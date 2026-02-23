from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import os
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QLineEdit, QSpinBox, QCheckBox, QMessageBox
)

from ..data.discover import discover_hdf5s
from ..data.index import MultiFileIndex
from ..data.h5io import list_feature_columns, read_images_by_indices, write_features_rows_inplace
from ..data.export import append_rows_to_h5
from ..imaging.render import channels_to_rgb8bit, _scale_to_u8_percentile, _gray_to_rgb, _downsample_nn_hwc
from .widgets.file_list import FileListWidget
from .widgets.gallery import GalleryWidget
from .widgets.label_editor import LabelEditor


def pd_to_float_safe(s) -> np.ndarray:
    try:
        return s.astype(float).to_numpy()
    except Exception:
        import pandas as pd
        return pd.to_numeric(s, errors="coerce").to_numpy()


class GalleryPane(QWidget):
    """A header + a GalleryWidget."""
    def __init__(self, title: str, tile_clicked_cb):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: 600;")
        lay.addWidget(self.title)

        self.gallery = GalleryWidget()
        self.gallery.tile_clicked.connect(tile_clicked_cb)
        lay.addWidget(self.gallery, 1)

    def set_title(self, title: str):
        self.title.setText(title)

    def set_layout(self, n_cols: int, tile_px: int):
        self.gallery.set_layout(n_cols=n_cols, tile_px=tile_px)

    def set_tiles(self, tiles: List[Dict]):
        self.gallery.set_tiles(tiles)

    def set_tile_label(self, h5_path: str, row_idx: int, new_label: str):
        self.gallery.set_tile_label(h5_path, row_idx, new_label)

    def tiles(self) -> List[Dict]:
        return self.gallery.tiles()


class AnnotateTab(QWidget):
    """
    Gallery tab:
      - Shows montage-style tiles directly in the gallery, controlled by channel checkboxes.
      - No separate montage panel.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.root_dir: Optional[str] = None
        self.files: List[Tuple[str, str]] = []
        self.selected_paths: List[str] = []
        self.feature_cols_cache: Dict[str, List[str]] = {}
        self._tile_cache: Dict[Tuple[str, int, Tuple[str, ...]], np.ndarray] = {}
        self._tile_cache_max = 20000  # optional cap

        self.index = MultiFileIndex()

        # UI labels
        self.labels = ["unlabeled", "cells", "junk"]
        self.active_label = "junk"

        self.label_col = cfg.default_label_col
        self.sort_col = cfg.default_score_col
        self.sort_asc = False
        
        # Pagination variables
        self.page = 0
        self.page_left = 0
        self.page_right = 0

        self._build_ui()

    # ---------------- label mapping ----------------

    def _ui_label_from_storage(self, v) -> str:
        """Convert stored label (0/1/NaN or strings) -> UI label string."""
        try:
            if v is None:
                return "unlabeled"
            if isinstance(v, float) and np.isnan(v):
                return "unlabeled"
        except Exception:
            pass

        # numeric cases
        try:
            fv = float(v)
            if np.isfinite(fv):
                if int(fv) == 1:
                    return "junk"
                if int(fv) == 0:
                    return "cells"
        except Exception:
            pass

        # string cases
        s = str(v).strip().lower()
        if s in ("", "none", "nan", "null"):
            return "unlabeled"
        if "junk" in s or s == "1":
            return "junk"
        if "cell" in s or s == "0":
            return "cells"
        if "unlab" in s:
            return "unlabeled"
        return "unlabeled"

    def _storage_label_from_ui(self, ui_label: str):
        """Convert UI label string -> stored numeric label (float with NaN for unlabeled)."""
        s = (ui_label or "").strip().lower()
        if "junk" in s:
            return 1.0
        if "cell" in s:
            return 0.0
        return np.nan

    # ---------------- UI ----------------

    def _build_ui(self):
        outer = QHBoxLayout(self)

        # LEFT: file selection
        left = QVBoxLayout()
        self.btn_pick_root = QPushButton("Select data root")
        self.btn_pick_root.clicked.connect(self.pick_root)
        left.addWidget(self.btn_pick_root)

        self.file_list = FileListWidget()
        self.file_list.selection_changed.connect(self.on_file_selection_changed)
        left.addWidget(self.file_list)

        outer.addLayout(left, 1)

        # RIGHT: controls + galleries
        right = QVBoxLayout()

        # ---- Controls row ----
        ctl = QHBoxLayout()

        self.label_col_box = QLineEdit(self.label_col)
        self.label_col_box.setPlaceholderText("label column (e.g., label or v_label)")
        self.label_col_box.editingFinished.connect(self.on_label_col_changed)

        self.sort_col_box = QComboBox()
        self.sort_col_box.currentTextChanged.connect(self.on_sort_changed)

        self.chk_asc = QCheckBox("Ascending")
        self.chk_asc.setChecked(False)
        self.chk_asc.stateChanged.connect(self.on_sort_dir_changed)

        self.chk_hard = QCheckBox("Hard cases (|score-0.5|)")
        self.chk_hard.setChecked(False)
        self.chk_hard.stateChanged.connect(self.on_sort_dir_changed)

        self.chk_dual = QCheckBox("Dual Top-N view")
        self.chk_dual.setChecked(False)
        self.chk_dual.stateChanged.connect(self.on_dual_changed)

        self.spin_topn = QSpinBox()
        self.spin_topn.setRange(10, 5000)
        self.spin_topn.setValue(80)
        self.spin_topn.valueChanged.connect(lambda: self.render_page())

        self.page_size = QSpinBox()
        self.page_size.setRange(20, 10000)
        self.page_size.setValue(self.cfg.page_size)
        self.page_size.valueChanged.connect(lambda: self.render_page())

        self.n_cols = QSpinBox()
        self.n_cols.setRange(2, 30)
        self.n_cols.setValue(self.cfg.n_cols)
        self.n_cols.valueChanged.connect(lambda: self._apply_gallery_layouts())

        self.btn_rebuild = QPushButton("Rebuild index")
        self.btn_rebuild.clicked.connect(self.rebuild_index)

        ctl.addWidget(QLabel("Label col:"))
        ctl.addWidget(self.label_col_box)
        ctl.addWidget(QLabel("Sort col:"))
        ctl.addWidget(self.sort_col_box)
        ctl.addWidget(self.chk_asc)
        ctl.addWidget(self.chk_hard)
        ctl.addWidget(self.chk_dual)
        ctl.addWidget(QLabel("N:"))
        ctl.addWidget(self.spin_topn)
        ctl.addWidget(QLabel("Page size:"))
        ctl.addWidget(self.page_size)
        ctl.addWidget(QLabel("Cols:"))
        ctl.addWidget(self.n_cols)
        ctl.addWidget(self.btn_rebuild)
        right.addLayout(ctl)

        # ---- Channel selection row ----
        chan = QHBoxLayout()
        chan.addWidget(QLabel("Tile channels:"))

        self.chk_comp = QCheckBox("Composite")
        self.chk_dapi = QCheckBox("DAPI")
        self.chk_tritc = QCheckBox("TRITC")
        self.chk_fitc = QCheckBox("FITC")
        self.chk_cy5 = QCheckBox("CY5")

        # default: composite only
        self.chk_comp.setChecked(True)
        for cb in [self.chk_comp, self.chk_dapi, self.chk_tritc, self.chk_fitc, self.chk_cy5]:
            cb.stateChanged.connect(self.on_channel_selection_changed)
            chan.addWidget(cb)

        chan.addStretch(1)
        right.addLayout(chan)

        # ---- Navigation row ----
        nav = QHBoxLayout()
        
        # Single mode nav
        self.btn_prev = QPushButton("Prev")
        self.btn_next = QPushButton("Next")
        self.lbl_page = QLabel("Page 0")
        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next.clicked.connect(self.next_page)

        # Dual mode nav - LEFT
        self.btn_prev_left = QPushButton("Prev Left")
        self.btn_next_left = QPushButton("Next Left")
        self.lbl_page_left = QLabel("L Page 0")
        self.btn_prev_left.clicked.connect(self.prev_page_left)
        self.btn_next_left.clicked.connect(self.next_page_left)

        # Dual mode nav - RIGHT
        self.btn_prev_right = QPushButton("Prev Right")
        self.btn_next_right = QPushButton("Next Right")
        self.lbl_page_right = QLabel("R Page 0")
        self.btn_prev_right.clicked.connect(self.prev_page_right)
        self.btn_next_right.clicked.connect(self.next_page_right)

        # Add all to layout
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.lbl_page)

        nav.addWidget(self.btn_prev_left)
        nav.addWidget(self.btn_next_left)
        nav.addWidget(self.lbl_page_left)

        nav.addWidget(self.btn_prev_right)
        nav.addWidget(self.btn_next_right)
        nav.addWidget(self.lbl_page_right)
        
        nav.addStretch(1)

        # Select-all buttons
        self.btn_select_all = QPushButton("Select all (this page) → active label")
        self.btn_select_all.clicked.connect(self.select_all_current_page)

        self.btn_select_all_left = QPushButton("Select all LEFT → active label")
        self.btn_select_all_left.clicked.connect(self.select_all_left_page)

        self.btn_select_all_right = QPushButton("Select all RIGHT → active label")
        self.btn_select_all_right.clicked.connect(self.select_all_right_page)
        
        # Label Entire button
        self.btn_label_entire = QPushButton("Label entire selected HDF5s → active label")
        self.btn_label_entire.clicked.connect(self.label_entire_selected_files)

        self.btn_save = QPushButton("Save annotations (writes to features)")
        self.btn_save.clicked.connect(self.save_current_view)

        # Replace export with rebuild
        self.btn_rebuild_vidur = QPushButton("Rebuild vidur_junk / vidur_cells from labels")
        self.btn_rebuild_vidur.clicked.connect(self.rebuild_vidur_files_from_labels)

        nav.addWidget(self.btn_select_all)
        nav.addWidget(self.btn_select_all_left)
        nav.addWidget(self.btn_select_all_right)
        nav.addWidget(self.btn_label_entire)

        nav.addWidget(self.btn_save)
        nav.addWidget(self.btn_rebuild_vidur)
        right.addLayout(nav)

        # ---- Label editor ----
        self.label_editor = LabelEditor(self.labels, self.active_label)
        self.label_editor.active_label_changed.connect(self.on_active_label_changed)
        self.label_editor.labels_changed.connect(self.on_labels_changed)
        right.addWidget(self.label_editor)

        # ---- Galleries (single OR dual) ----
        self.single_pane = GalleryPane("Gallery", self.on_tile_clicked)
        self.dual_left_pane = GalleryPane("Top junk (high score)", self.on_tile_clicked)
        self.dual_right_pane = GalleryPane("Top cells (low score)", self.on_tile_clicked)

        self.gal_row = QHBoxLayout()
        self.gal_row.addWidget(self.single_pane, 1)
        self.gal_row.addWidget(self.dual_left_pane, 1)
        self.gal_row.addWidget(self.dual_right_pane, 1)
        right.addLayout(self.gal_row, 3)

        outer.addLayout(right, 3)

        self._apply_gallery_layouts()
        self._sync_gallery_visibility()

    def _apply_gallery_layouts(self):
        n_cols = int(self.n_cols.value())
        k = len(self._selected_tiles_spec())
        tile_h = int(self.cfg.tile_px)
        tile_w = int(tile_h * k)

        self.single_pane.gallery.set_layout(n_cols=n_cols, tile_h=tile_h, tile_w=tile_w)
        self.dual_left_pane.gallery.set_layout(n_cols=n_cols, tile_h=tile_h, tile_w=tile_w)
        self.dual_right_pane.gallery.set_layout(n_cols=n_cols, tile_h=tile_h, tile_w=tile_w)

    def _sync_gallery_visibility(self):
        dual = self.chk_dual.isChecked()
        self.single_pane.setVisible(not dual)
        self.dual_left_pane.setVisible(dual)
        self.dual_right_pane.setVisible(dual)

        # Single nav visibility
        self.btn_prev.setVisible(not dual)
        self.btn_next.setVisible(not dual)
        self.lbl_page.setVisible(not dual)

        # Dual nav visibility
        self.btn_prev_left.setVisible(dual)
        self.btn_next_left.setVisible(dual)
        self.lbl_page_left.setVisible(dual)
        self.btn_prev_right.setVisible(dual)
        self.btn_next_right.setVisible(dual)
        self.lbl_page_right.setVisible(dual)

        # show correct select-all buttons
        self.btn_select_all.setVisible(not dual)
        self.btn_select_all_left.setVisible(dual)
        self.btn_select_all_right.setVisible(dual)

    # ---------------- Events ----------------

    def pick_root(self):
        default_dir = "/mnt/deepstore/Vidur/Junk_Classification/data"
        if not os.path.exists(default_dir):
            default_dir = str(Path.cwd())
            
        d = QFileDialog.getExistingDirectory(self, "Select a directory", default_dir)
        if not d:
            return

        selected_path = Path(d)
        parent_directory = selected_path.parent
        data_folder_path = parent_directory / "data"

        if not data_folder_path.is_dir():
            QMessageBox.warning(self, "Missing data folder", f"'data' directory not found at:\n{data_folder_path}")
            return

        self.root_dir = str(data_folder_path)
        self.files = discover_hdf5s(str(data_folder_path))
        self.file_list.set_files(self.files)

    def on_file_selection_changed(self, selected_paths: List[str]):
        self.selected_paths = selected_paths
        if not self.selected_paths:
            return

        p0 = self.selected_paths[0]
        cols = self.feature_cols_cache.get(p0)
        if cols is None:
            cols = list_feature_columns(p0, features_key=self.cfg.features_key)
            self.feature_cols_cache[p0] = cols

        self.sort_col_box.blockSignals(True)
        self.sort_col_box.clear()
        self.sort_col_box.addItems(cols)
        if self.sort_col in cols:
            self.sort_col_box.setCurrentText(self.sort_col)
        self.sort_col_box.blockSignals(False)

        self.rebuild_index()

    def on_label_col_changed(self):
        self.label_col = self.label_col_box.text().strip() or self.cfg.default_label_col
        self.rebuild_index()

    def on_sort_changed(self, col: str):
        if not col:
            return
        self.sort_col = col
        self.rebuild_index()

    def on_sort_dir_changed(self):
        self.sort_asc = self.chk_asc.isChecked()
        self.rebuild_index()

    def on_dual_changed(self):
        self.page = 0
        self.page_left = 0
        self.page_right = 0
        self._sync_gallery_visibility()
        self.render_page()

    def on_active_label_changed(self, name: str):
        self.active_label = name

    def on_labels_changed(self, labels: List[str]):
        self.labels = labels

    def on_channel_selection_changed(self):
        if not any([self.chk_comp.isChecked(), self.chk_dapi.isChecked(),
                    self.chk_tritc.isChecked(), self.chk_fitc.isChecked(), self.chk_cy5.isChecked()]):
            self.chk_comp.blockSignals(True)
            self.chk_comp.setChecked(True)
            self.chk_comp.blockSignals(False)

        self._apply_gallery_layouts()
        self.render_page()

    # ---------------- Index build/sort ----------------

    def rebuild_index(self):
        if not self.selected_paths:
            return

        needed = list({self.sort_col, self.label_col, "size", "slide_id", "frame_id", "cell_id"})
        self.index.build(self.selected_paths, needed_cols=needed, features_key=self.cfg.features_key)

        if self.chk_hard.isChecked():
            self._sort_hard_cases_inplace(self.sort_col)
        else:
            self.index.sort(self.sort_col, ascending=self.sort_asc)

        self.page = 0
        self.render_page()

    def _sort_hard_cases_inplace(self, score_col: str):
        if self.index.table is None or len(self.index.table) == 0:
            return
        if score_col not in self.index.table.columns:
            return

        df = self.index.table
        s = pd_to_float_safe(df[score_col])
        df["__hard__"] = np.abs(s - 0.5)
        df.sort_values("__hard__", ascending=True, inplace=True, kind="mergesort")
        df.drop(columns=["__hard__"], inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)

    # ---------------- Tile rendering (montage in-gallery) ----------------

    def _selected_tiles_spec(self) -> List[Tuple[str, Optional[int]]]:
        """
        Returns list of (name, channel_index or None for composite) in display order.
        Assumed channel indices:
          0=DAPI, 1=TRITC, 2=CY5, 3=FITC
        """
        out: List[Tuple[str, Optional[int]]] = []
        if self.chk_comp.isChecked():
            out.append(("Composite", None))
        if self.chk_dapi.isChecked():
            out.append(("DAPI", 0))
        if self.chk_tritc.isChecked():
            out.append(("TRITC", 1))
        if self.chk_fitc.isChecked():
            out.append(("FITC", 3))
        if self.chk_cy5.isChecked():
            out.append(("CY5", 2))
        if not out:
            out = [("Composite", None)]
        return out

    def _render_tile_rgb(self, img_hwc: np.ndarray) -> np.ndarray:
        img = np.asarray(img_hwc)
        _, _, c = img.shape
        spec = self._selected_tiles_spec()

        tile_h = int(self.cfg.tile_px)
        k = max(1, len(spec))
        tile_w = tile_h * k
        panel_w = tile_w // k

        panels = []
        for _, ch in spec:
            if ch is None:
                comp = channels_to_rgb8bit(img)  # HxWx3 uint8
                comp_s = _downsample_nn_hwc(comp, tile_h, panel_w)
                panels.append(comp_s)
            else:
                if ch >= c:
                    panels.append(np.zeros((tile_h, panel_w, 3), dtype=np.uint8))
                else:
                    g = _scale_to_u8_percentile(img[..., ch])
                    rgb = _gray_to_rgb(g)
                    rgb_s = _downsample_nn_hwc(rgb, tile_h, panel_w)
                    panels.append(rgb_s)

        return np.concatenate(panels, axis=1).astype(np.uint8, copy=False)

    # ---------------- Rendering ----------------

    def render_page(self):
        n = self.index.size()
        if n == 0:
            self.single_pane.set_tiles([])
            self.dual_left_pane.set_tiles([])
            self.dual_right_pane.set_tiles([])
            self.lbl_page.setText("Empty")
            return

        if self.chk_dual.isChecked():
            self._render_dual_topn()
            return

        ps = int(self.page_size.value())
        start = self.page * ps
        start = max(0, min(start, max(0, n - ps)))
        self.page = start // ps

        view = self.index.page(start, ps)
        self.lbl_page.setText(f"Page {self.page}  (rows {start+1}-{min(start+ps, n)} / {n})")

        tiles = self._tiles_from_view(view)
        self._apply_gallery_layouts()
        self.single_pane.set_tiles(tiles)

    def _render_dual_topn(self):
        if self.sort_col not in self.index.table.columns:
            self.dual_left_pane.set_tiles([])
            self.dual_right_pane.set_tiles([])
            self.lbl_page_left.setText(f"Error: missing sort col '{self.sort_col}'")
            self.lbl_page_right.setText(f"Error: missing sort col '{self.sort_col}'")
            return

        N = int(self.spin_topn.value())
        ps = int(self.page_size.value()) # Get page size
        
        df = self.index.table.copy()
        score = pd_to_float_safe(df[self.sort_col])
        df = df[np.isfinite(score)].copy()
        if len(df) == 0:
            self.dual_left_pane.set_tiles([])
            self.dual_right_pane.set_tiles([])
            self.lbl_page_left.setText("Dual view: no finite scores")
            self.lbl_page_right.setText("Dual view: no finite scores")
            return

        top = df.nlargest(N, self.sort_col)
        bot = df.nsmallest(N, self.sort_col)

        # --- LEFT PANE PAGINATION ---
        start_left = self.page_left * ps
        start_left = max(0, min(start_left, max(0, len(top) - ps)))
        self.page_left = start_left // ps
        view_top = top.iloc[start_left : start_left + ps]
        self.lbl_page_left.setText(f"L Page {self.page_left} ({start_left+1}-{min(start_left+ps, len(top))}/{len(top)})")

        # --- RIGHT PANE PAGINATION ---
        start_right = self.page_right * ps
        start_right = max(0, min(start_right, max(0, len(bot) - ps)))
        self.page_right = start_right // ps
        view_bot = bot.iloc[start_right : start_right + ps]
        self.lbl_page_right.setText(f"R Page {self.page_right} ({start_right+1}-{min(start_right+ps, len(bot))}/{len(bot)})")

        self.dual_left_pane.set_title(f"Top junk (high {self.sort_col})")
        self.dual_right_pane.set_title(f"Top cells (low {self.sort_col})")

        tiles_top = self._tiles_from_view(view_top)
        tiles_bot = self._tiles_from_view(view_bot)

        self._apply_gallery_layouts()
        self.dual_left_pane.set_tiles(tiles_top)
        self.dual_right_pane.set_tiles(tiles_bot)

    def _tiles_from_view(self, view_df) -> List[Dict]:
        if view_df is None or len(view_df) == 0:
            return []

        mask = tuple(name for name, _ in self._selected_tiles_spec())

        by_file: Dict[str, List[int]] = {}
        for _, r in view_df.iterrows():
            by_file.setdefault(r["__h5_path__"], []).append(int(r["__row_idx__"]))

        for fp, idxs in by_file.items():
            idxs_int = [int(x) for x in idxs]
            missing = [ridx for ridx in idxs_int if (fp, ridx, mask) not in self._tile_cache]
            if not missing:
                continue

            idxs_sorted = np.asarray(sorted(missing), dtype=int)
            imgs = read_images_by_indices(fp, idxs_sorted, image_key=self.cfg.image_key)

            for ridx, im in zip(idxs_sorted.tolist(), imgs):
                key = (fp, int(ridx), mask)
                self._tile_cache[key] = self._render_tile_rgb(im)

            if getattr(self, "_tile_cache_max", None) is not None:
                maxn = int(self._tile_cache_max)
                if len(self._tile_cache) > maxn:
                    for k in list(self._tile_cache.keys())[: len(self._tile_cache) - maxn]:
                        self._tile_cache.pop(k, None)

        tiles: List[Dict] = []
        for _, r in view_df.iterrows():
            fp = r["__h5_path__"]
            ridx = int(r["__row_idx__"])

            stored_label = r.get(self.label_col, np.nan)
            ui_label = self._ui_label_from_storage(stored_label)

            base = os.path.basename(fp)
            tooltip = (
                f"{base} | idx={ridx}\n"
                f"score({self.sort_col})={r.get(self.sort_col)}\n"
                f"label({self.label_col})={stored_label}"
            )

            key = (fp, ridx, mask)
            rgb = self._tile_cache.get(key)
            if rgb is None:
                tile_h = int(self.cfg.tile_px)
                k = max(1, len(mask))
                tile_w = tile_h * k
                rgb = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

            tiles.append({
                "h5_path": fp,
                "row_idx": ridx,
                "rgb": rgb,
                "label": ui_label,          # UI label string
                "tooltip": tooltip,
            })

        return tiles

    # ---------------- Paging ----------------

    def next_page(self):
        if self.chk_dual.isChecked():
            return
        self.page += 1
        self.render_page()

    def prev_page(self):
        if self.chk_dual.isChecked():
            return
        self.page = max(0, self.page - 1)
        self.render_page()

    def next_page_left(self):
        self.page_left += 1
        self.render_page()

    def prev_page_left(self):
        self.page_left = max(0, self.page_left - 1)
        self.render_page()

    def next_page_right(self):
        self.page_right += 1
        self.render_page()

    def prev_page_right(self):
        self.page_right = max(0, self.page_right - 1)
        self.render_page()

    # ---------------- Click ----------------

    def on_tile_clicked(self, h5_path: str, row_idx: int, button: str):
        new_label = self.active_label if button == "left" else "unlabeled"
        self.single_pane.set_tile_label(h5_path, row_idx, new_label)
        self.dual_left_pane.set_tile_label(h5_path, row_idx, new_label)
        self.dual_right_pane.set_tile_label(h5_path, row_idx, new_label)

    # ---------------- Select All & Label Entire File ----------------

    def select_all_current_page(self):
        """Single view: set all visible tiles to active label."""
        tiles = list(self.single_pane.tiles() or [])
        for t in tiles:
            self.single_pane.set_tile_label(t["h5_path"], int(t["row_idx"]), self.active_label)

    def select_all_left_page(self):
        """Dual view: set all LEFT visible tiles to active label."""
        tiles = list(self.dual_left_pane.tiles() or [])
        for t in tiles:
            self.dual_left_pane.set_tile_label(t["h5_path"], int(t["row_idx"]), self.active_label)

    def select_all_right_page(self):
        """Dual view: set all RIGHT visible tiles to active label."""
        tiles = list(self.dual_right_pane.tiles() or [])
        for t in tiles:
            self.dual_right_pane.set_tile_label(t["h5_path"], int(t["row_idx"]), self.active_label)

    def label_entire_selected_files(self):
        """Apply the active label to EVERY row in the currently selected HDF5 files."""
        if not self.selected_paths:
            QMessageBox.information(self, "No files", "Select at least one HDF5 file in the file list on the left.")
            return

        reply = QMessageBox.question(
            self, "Confirm",
            f"Set ALL rows in {len(self.selected_paths)} selected file(s) to '{self.active_label}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        v = self._storage_label_from_ui(self.active_label)
        val = float(v) if np.isfinite(v) else np.nan

        import h5py
        from ..data.h5io import write_features_column_inplace
        
        count = 0
        for fp in self.selected_paths:
            try:
                # Use the images block to figure out the row count
                with h5py.File(fp, "r") as f:
                    n = f[self.cfg.image_key].shape[0]
                arr = np.full(n, val, dtype=np.float32)
                write_features_column_inplace(fp, self.label_col, arr, features_key=self.cfg.features_key)
                count += 1
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to label {os.path.basename(fp)}:\n{e}")
        
        QMessageBox.information(self, "Done", f"Labeled {count} file(s) entirely as '{self.active_label}'.")
        self.rebuild_index()

    # ---------------- Save / Rebuild ----------------

    def _current_tiles(self) -> List[Dict]:
        if self.chk_dual.isChecked():
            return list(self.dual_left_pane.tiles() or []) + list(self.dual_right_pane.tiles() or [])
        return list(self.single_pane.tiles() or [])

    def save_current_view(self):
        tiles = self._current_tiles()
        if not tiles:
            return

        # write numeric labels: junk=1, cells=0, unlabeled=NaN
        by_file: Dict[str, List[Tuple[int, float]]] = {}
        for t in tiles:
            fp = t["h5_path"]
            ridx = int(t["row_idx"])
            v = self._storage_label_from_ui(str(t["label"]))
            by_file.setdefault(fp, []).append((ridx, float(v) if np.isfinite(v) else np.nan))

        try:
            for fp, rows in by_file.items():
                idx = np.array([r for r, _ in rows], dtype=int)
                vals = np.array([v for _, v in rows], dtype=np.float32)
                write_features_rows_inplace(fp, idx, self.label_col, vals, features_key=self.cfg.features_key)

            QMessageBox.information(self, "Saved", f"Wrote numeric '{self.label_col}' for {len(by_file)} file(s).")
            # refresh view so label strings reflect stored numeric values consistently
            self.rebuild_index()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def rebuild_vidur_files_from_labels(self):
        """
        Reconstruct vidur_junk.hdf5 and vidur_cells.hdf5 by scanning labels
        in the currently selected HDF5 files.
        """
        if not self.root_dir:
            QMessageBox.information(self, "No root", "Select a root directory first.")
            return

        if not self.selected_paths:
            QMessageBox.information(self, "No files", "Select one or more HDF5 files from the file list.")
            return

        junk_dir = os.path.join(self.root_dir, self.cfg.export_junk_dirname)
        cell_dir = os.path.join(self.root_dir, self.cfg.export_cells_dirname)
        os.makedirs(junk_dir, exist_ok=True)
        os.makedirs(cell_dir, exist_ok=True)

        dst_junk = os.path.join(junk_dir, "vidur_junk.hdf5")
        dst_cell = os.path.join(cell_dir, "vidur_cells.hdf5")

        # delete old outputs so it's a true rebuild
        for p in [dst_junk, dst_cell]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        import pandas as pd

        n_junk_total = 0
        n_cell_total = 0

        for fp in self.selected_paths:
            try:
                df = pd.read_hdf(fp, key=self.cfg.features_key)
            except Exception:
                continue
            if self.label_col not in df.columns:
                continue

            col = df[self.label_col]

            # handle both numeric and legacy string labels
            junk_mask = None
            cell_mask = None
            if np.issubdtype(col.dtype, np.number):
                junk_mask = (col == 1)
                cell_mask = (col == 0)
            else:
                s = col.astype(str).str.lower()
                junk_mask = s.str.contains("junk") | (s == "1")
                cell_mask = s.str.contains("cell") | (s == "0")

            junk_idx = np.where(junk_mask.to_numpy())[0].astype(int)
            cell_idx = np.where(cell_mask.to_numpy())[0].astype(int)

            if len(junk_idx) > 0:
                append_rows_to_h5(fp, junk_idx, dst_junk,
                                  image_key=self.cfg.image_key, features_key=self.cfg.features_key)
                n_junk_total += int(len(junk_idx))

            if len(cell_idx) > 0:
                append_rows_to_h5(fp, cell_idx, dst_cell,
                                  image_key=self.cfg.image_key, features_key=self.cfg.features_key)
                n_cell_total += int(len(cell_idx))

        QMessageBox.information(
            self,
            "Rebuilt",
            f"Rebuilt:\n  {dst_junk}  (n={n_junk_total})\n  {dst_cell}  (n={n_cell_total})"
        )