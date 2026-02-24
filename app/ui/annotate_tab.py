# app/ui/annotate_tab.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, 
    QComboBox, QLineEdit, QSpinBox, QCheckBox, QMessageBox
)
from PyQt5.QtCore import QSettings

from ..data.discover import discover_hdf5s
from ..data.index import MultiFileIndex
from ..data.h5io import list_feature_columns, read_images_by_indices, write_features_rows_inplace
from ..data.batch_ops import batch_label_files, rebuild_vidur_files
from ..imaging.render import channels_to_rgb8bit, _scale_to_u8_percentile, _gray_to_rgb, _downsample_nn_hwc
from .widgets.file_list import FileListWidget
from .widgets.label_editor import LabelEditor
# FIX: Import GalleryPane
from .widgets.gallery_pane import GalleryPane

def pd_to_float_safe(s):
    try: return s.astype(float).to_numpy()
    except: 
        return pd.to_numeric(s, errors="coerce").to_numpy()

class AnnotateTab(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.settings = QSettings("VidurLab", "JunkAnnotator")
        self.root_dir = None
        self.files = []
        self.selected_paths = []
        self.feature_cols_cache = {}
        self._tile_cache = {}
        self.index = MultiFileIndex()
        self.labels = ["unlabeled", "cells", "junk"]
        self.active_label = "junk"
        self.label_col = cfg.default_label_col
        self.sort_col = cfg.default_score_col
        self.sort_asc = False
        self.page = 0
        self.page_left = 0
        self.page_right = 0
        self._build_ui()
        
        last_root = self.settings.value("last_root_dir", "")
        if last_root and os.path.isdir(last_root):
            self.set_root_dir(last_root)

    def _ui_label_from_storage(self, v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)): return "unlabeled"
            fv = float(v)
            if np.isfinite(fv):
                return "junk" if int(fv) == 1 else "cells"
        except: pass
        s = str(v).strip().lower()
        if s in ("", "nan", "none"): return "unlabeled"
        if "junk" in s or s == "1": return "junk"
        if "cell" in s or s == "0": return "cells"
        return "unlabeled"

    def _storage_label_from_ui(self, ui):
        s = (ui or "").strip().lower()
        if "junk" in s: return 1.0
        if "cell" in s: return 0.0
        return np.nan

    def _build_ui(self):
        outer = QHBoxLayout(self)
        left = QVBoxLayout()
        self.btn_pick_root = QPushButton("Select data root")
        self.btn_pick_root.clicked.connect(self.pick_root)
        left.addWidget(self.btn_pick_root)
        self.file_list = FileListWidget()
        self.file_list.selection_changed.connect(self.on_file_selection_changed)
        left.addWidget(self.file_list)
        outer.addLayout(left, 1)

        right = QVBoxLayout()
        ctl = QHBoxLayout()
        self.label_col_box = QLineEdit(self.label_col)
        self.label_col_box.editingFinished.connect(self.on_label_col_changed)
        self.sort_col_box = QComboBox()
        self.sort_col_box.currentTextChanged.connect(self.on_sort_changed)
        self.chk_asc = QCheckBox("Ascending")
        self.chk_asc.stateChanged.connect(self.on_sort_dir_changed)
        self.chk_hard = QCheckBox("Hard cases")
        self.chk_hard.stateChanged.connect(self.on_sort_dir_changed)
        self.chk_dual = QCheckBox("Dual Top-N")
        self.chk_dual.stateChanged.connect(self.on_dual_changed)
        self.spin_topn = QSpinBox(); self.spin_topn.setRange(10, 5000); self.spin_topn.setValue(80)
        self.spin_topn.valueChanged.connect(lambda: self.render_page())
        self.page_size = QSpinBox(); self.page_size.setRange(20, 10000); self.page_size.setValue(self.cfg.page_size)
        self.page_size.valueChanged.connect(lambda: self.render_page())
        self.n_cols = QSpinBox(); self.n_cols.setRange(2, 30); self.n_cols.setValue(self.cfg.n_cols)
        self.n_cols.valueChanged.connect(lambda: self._apply_gallery_layouts())
        self.btn_rebuild = QPushButton("Rebuild index")
        self.btn_rebuild.clicked.connect(self.rebuild_index)

        ctl.addWidget(QLabel("Label col:")); ctl.addWidget(self.label_col_box)
        ctl.addWidget(QLabel("Sort col:")); ctl.addWidget(self.sort_col_box)
        ctl.addWidget(self.chk_asc); ctl.addWidget(self.chk_hard); ctl.addWidget(self.chk_dual)
        ctl.addWidget(QLabel("N:")); ctl.addWidget(self.spin_topn)
        ctl.addWidget(QLabel("Page size:")); ctl.addWidget(self.page_size)
        ctl.addWidget(QLabel("Cols:")); ctl.addWidget(self.n_cols)
        ctl.addWidget(self.btn_rebuild)
        right.addLayout(ctl)

        chan = QHBoxLayout()
        chan.addWidget(QLabel("Tile channels:"))
        self.chk_comp = QCheckBox("Composite"); self.chk_comp.setChecked(True)
        self.chk_dapi = QCheckBox("DAPI")
        self.chk_tritc = QCheckBox("TRITC")
        self.chk_fitc = QCheckBox("FITC")
        self.chk_cy5 = QCheckBox("CY5")
        for cb in [self.chk_comp, self.chk_dapi, self.chk_tritc, self.chk_fitc, self.chk_cy5]:
            cb.stateChanged.connect(self.on_channel_selection_changed)
            chan.addWidget(cb)
        chan.addStretch(1)
        right.addLayout(chan)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("Prev"); self.btn_next = QPushButton("Next")
        self.lbl_page = QLabel("Page 0")
        self.btn_prev.clicked.connect(self.prev_page); self.btn_next.clicked.connect(self.next_page)
        
        self.btn_prev_left = QPushButton("Prev Left"); self.btn_next_left = QPushButton("Next Left")
        self.lbl_page_left = QLabel("L Page 0")
        self.btn_prev_left.clicked.connect(self.prev_page_left); self.btn_next_left.clicked.connect(self.next_page_left)

        self.btn_prev_right = QPushButton("Prev Right"); self.btn_next_right = QPushButton("Next Right")
        self.lbl_page_right = QLabel("R Page 0")
        self.btn_prev_right.clicked.connect(self.prev_page_right); self.btn_next_right.clicked.connect(self.next_page_right)

        nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next); nav.addWidget(self.lbl_page)
        nav.addWidget(self.btn_prev_left); nav.addWidget(self.btn_next_left); nav.addWidget(self.lbl_page_left)
        nav.addWidget(self.btn_prev_right); nav.addWidget(self.btn_next_right); nav.addWidget(self.lbl_page_right)
        nav.addStretch(1)

        self.btn_select_all = QPushButton("Select all (page) → active")
        self.btn_select_all.clicked.connect(self.select_all_current_page)
        self.btn_select_all_left = QPushButton("Select all LEFT → active")
        self.btn_select_all_left.clicked.connect(self.select_all_left_page)
        self.btn_select_all_right = QPushButton("Select all RIGHT → active")
        self.btn_select_all_right.clicked.connect(self.select_all_right_page)
        
        self.btn_label_entire = QPushButton("Label entire files → active")
        self.btn_label_entire.clicked.connect(self.label_entire_selected_files)
        self.btn_save = QPushButton("Save annotations")
        self.btn_save.clicked.connect(self.save_current_view)
        self.btn_rebuild_vidur = QPushButton("Rebuild vidur_junk/cells")
        self.btn_rebuild_vidur.clicked.connect(self.rebuild_vidur_files_from_labels)

        nav.addWidget(self.btn_select_all); nav.addWidget(self.btn_select_all_left); nav.addWidget(self.btn_select_all_right)
        nav.addWidget(self.btn_label_entire); nav.addWidget(self.btn_save); nav.addWidget(self.btn_rebuild_vidur)
        right.addLayout(nav)

        self.label_editor = LabelEditor(self.labels, self.active_label)
        self.label_editor.active_label_changed.connect(self.on_active_label_changed)
        self.label_editor.labels_changed.connect(self.on_labels_changed)
        right.addWidget(self.label_editor)

        self.single_pane = GalleryPane("Gallery", self.on_tile_clicked)
        self.dual_left_pane = GalleryPane("Top junk", self.on_tile_clicked)
        self.dual_right_pane = GalleryPane("Top cells", self.on_tile_clicked)
        self.gal_row = QHBoxLayout()
        self.gal_row.addWidget(self.single_pane, 1)
        self.gal_row.addWidget(self.dual_left_pane, 1)
        self.gal_row.addWidget(self.dual_right_pane, 1)
        right.addLayout(self.gal_row, 3)
        outer.addLayout(right, 3)
        
        self._apply_gallery_layouts()
        self._sync_gallery_visibility()

    def pick_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select a directory", "/mnt/deepstore/Vidur/Junk_Classification/data")
        if not d: return
        p = Path(d); data = p.parent / "data"
        if not data.is_dir(): return QMessageBox.warning(self, "Missing data", f"No data at {data}")
        self.set_root_dir(str(data))

    def set_root_dir(self, path):
        self.root_dir = path
        self.settings.setValue("last_root_dir", path)
        self.files = discover_hdf5s(path)
        self.file_list.set_files(self.files)

    def on_file_selection_changed(self, paths):
        self.selected_paths = paths
        if not paths: return
        cols = self.feature_cols_cache.get(paths[0])
        if not cols:
            cols = list_feature_columns(paths[0], features_key=self.cfg.features_key)
            self.feature_cols_cache[paths[0]] = cols
        self.sort_col_box.blockSignals(True); self.sort_col_box.clear(); self.sort_col_box.addItems(cols)
        if self.sort_col in cols: self.sort_col_box.setCurrentText(self.sort_col)
        self.sort_col_box.blockSignals(False)
        self.rebuild_index()

    def rebuild_index(self):
        if not self.selected_paths: return
        needed = list({self.sort_col, self.label_col, "size", "slide_id", "frame_id", "cell_id"})
        self.index.build(self.selected_paths, needed_cols=needed, features_key=self.cfg.features_key)
        if self.chk_hard.isChecked(): self._sort_hard_cases_inplace(self.sort_col)
        else: self.index.sort(self.sort_col, ascending=self.sort_asc)
        self.page = 0; self.render_page()

    def _sort_hard_cases_inplace(self, sc):
        if self.index.table is None or sc not in self.index.table.columns: return
        df = self.index.table
        df["__hard__"] = np.abs(pd_to_float_safe(df[sc]) - 0.5)
        df.sort_values("__hard__", ascending=True, inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)

    def _render_tile_rgb(self, img):
        spec = self._selected_tiles_spec()
        tile_h = int(self.cfg.tile_px); k = max(1, len(spec)); panel_w = tile_h
        panels = []
        for _, ch in spec:
            if ch is None: panels.append(_downsample_nn_hwc(channels_to_rgb8bit(img), tile_h, panel_w))
            else:
                if ch >= img.shape[2]: panels.append(np.zeros((tile_h, panel_w, 3), dtype=np.uint8))
                else: panels.append(_downsample_nn_hwc(_gray_to_rgb(_scale_to_u8_percentile(img[..., ch])), tile_h, panel_w))
        return np.concatenate(panels, axis=1)

    def _selected_tiles_spec(self):
        out = []
        if self.chk_comp.isChecked(): out.append(("Composite", None))
        if self.chk_dapi.isChecked(): out.append(("DAPI", 0))
        if self.chk_tritc.isChecked(): out.append(("TRITC", 1))
        if self.chk_fitc.isChecked(): out.append(("FITC", 3))
        if self.chk_cy5.isChecked(): out.append(("CY5", 2))
        return out or [("Composite", None)]

    def render_page(self):
        if self.index.size() == 0:
            for p in [self.single_pane, self.dual_left_pane, self.dual_right_pane]: p.set_tiles([])
            return
        if self.chk_dual.isChecked(): self._render_dual(); return
        ps = int(self.page_size.value()); start = self.page * ps
        view = self.index.page(start, ps)
        self.lbl_page.setText(f"Page {self.page} ({start+1}-{min(start+ps, self.index.size())})")
        self.single_pane.set_tiles(self._tiles_from_view(view))
        self._apply_gallery_layouts()

    def _render_dual(self):
        N = int(self.spin_topn.value()); ps = int(self.page_size.value())
        df = self.index.table
        score = pd_to_float_safe(df[self.sort_col])
        df = df[np.isfinite(score)]
        top = df.nlargest(N, self.sort_col); bot = df.nsmallest(N, self.sort_col)
        
        sl = self.page_left * ps; vl = top.iloc[sl:sl+ps]
        self.lbl_page_left.setText(f"L Page {self.page_left} ({sl+1}-{sl+len(vl)})")
        self.dual_left_pane.set_tiles(self._tiles_from_view(vl))
        
        sr = self.page_right * ps; vr = bot.iloc[sr:sr+ps]
        self.lbl_page_right.setText(f"R Page {self.page_right} ({sr+1}-{sr+len(vr)})")
        self.dual_right_pane.set_tiles(self._tiles_from_view(vr))
        self._apply_gallery_layouts()

    def _tiles_from_view(self, df):
        if df is None or len(df) == 0: return []
        mask = tuple(n for n, _ in self._selected_tiles_spec())
        by_file = {}
        for _, r in df.iterrows(): by_file.setdefault(r["__h5_path__"], []).append(int(r["__row_idx__"]))
        
        for fp, idxs in by_file.items():
            missing = [i for i in idxs if (fp, i, mask) not in self._tile_cache]
            if missing:
                imgs = read_images_by_indices(fp, np.array(sorted(missing)), image_key=self.cfg.image_key)
                for i, im in zip(sorted(missing), imgs): self._tile_cache[(fp, i, mask)] = self._render_tile_rgb(im)
                
        tiles = []
        for _, r in df.iterrows():
            fp = r["__h5_path__"]; idx = int(r["__row_idx__"])
            tiles.append({
                "h5_path": fp, "row_idx": idx, 
                "rgb": self._tile_cache.get((fp, idx, mask)),
                "label": self._ui_label_from_storage(r.get(self.label_col)),
                "tooltip": f"{os.path.basename(fp)}|{idx}\n{r.get(self.sort_col)}"
            })
        return tiles

    # Handlers
    def on_label_col_changed(self): self.label_col = self.label_col_box.text() or self.cfg.default_label_col; self.rebuild_index()
    def on_sort_changed(self, c): self.sort_col = c; self.rebuild_index()
    def on_sort_dir_changed(self): self.sort_asc = self.chk_asc.isChecked(); self.rebuild_index()
    def on_dual_changed(self): self._sync_gallery_visibility(); self.render_page()
    def on_active_label_changed(self, n): self.active_label = n
    def on_labels_changed(self, l): self.labels = l
    def on_channel_selection_changed(self): self._apply_gallery_layouts(); self.render_page()
    def on_tile_clicked(self, p, r, b):
        l = self.active_label if b == "left" else "unlabeled"
        for pane in [self.single_pane, self.dual_left_pane, self.dual_right_pane]: pane.set_tile_label(p, r, l)
    
    def _apply_gallery_layouts(self):
        nc = int(self.n_cols.value()); th = int(self.cfg.tile_px); tw = th * len(self._selected_tiles_spec())
        for p in [self.single_pane, self.dual_left_pane, self.dual_right_pane]: 
            p.set_layout(n_cols=nc, tile_h=th, tile_w=tw)

    def _sync_gallery_visibility(self):
        d = self.chk_dual.isChecked()
        self.single_pane.setVisible(not d); self.dual_left_pane.setVisible(d); self.dual_right_pane.setVisible(d)
        self.btn_select_all.setVisible(not d); self.btn_select_all_left.setVisible(d); self.btn_select_all_right.setVisible(d)

    # File Ops
    def label_entire_selected_files(self):
        if not self.selected_paths: return QMessageBox.info(self, "No files", "Select files.")
        if QMessageBox.question(self, "Confirm", f"Label {len(self.selected_paths)} files as '{self.active_label}'?", QMessageBox.Yes|QMessageBox.No) != QMessageBox.Yes: return
        val = self._storage_label_from_ui(self.active_label)
        v = float(val) if np.isfinite(val) else np.nan
        cnt, errs = batch_label_files(self.selected_paths, self.label_col, v, self.cfg.image_key, self.cfg.features_key)
        if errs: QMessageBox.warning(self, "Errors", "\n".join(errs))
        QMessageBox.information(self, "Done", f"Labeled {cnt} files.")
        self.rebuild_index()

    def save_current_view(self):
        tiles = self._current_tiles()
        if not tiles: return
        by_file = {}
        for t in tiles:
            v = self._storage_label_from_ui(t["label"])
            by_file.setdefault(t["h5_path"], []).append((t["row_idx"], float(v) if np.isfinite(v) else np.nan))
        try:
            for fp, rows in by_file.items():
                write_features_rows_inplace(fp, [r for r,_ in rows], self.label_col, [v for _,v in rows], features_key=self.cfg.features_key)
            QMessageBox.information(self, "Saved", f"Saved {len(by_file)} files.")
            self.rebuild_index()
        except Exception as e: QMessageBox.warning(self, "Error", str(e))

    def rebuild_vidur_files_from_labels(self):
        if not self.root_dir or not self.selected_paths: return QMessageBox.information(self, "Info", "Select root and files.")
        try:
            j, nj, c, nc = rebuild_vidur_files(self.selected_paths, self.root_dir, self.label_col, self.cfg.export_junk_dirname, self.cfg.export_cells_dirname, self.cfg.image_key, self.cfg.features_key)
            QMessageBox.information(self, "Rebuilt", f"Junk: {nj}\nCells: {nc}")
        except Exception as e: QMessageBox.warning(self, "Error", str(e))

    # Nav helpers
    def prev_page(self): self.page = max(0, self.page-1); self.render_page()
    def next_page(self): self.page += 1; self.render_page()
    def prev_page_left(self): self.page_left = max(0, self.page_left-1); self.render_page()
    def next_page_left(self): self.page_left += 1; self.render_page()
    def prev_page_right(self): self.page_right = max(0, self.page_right-1); self.render_page()
    def next_page_right(self): self.page_right += 1; self.render_page()
    def select_all_current_page(self): self._set_all(self.single_pane.tiles())
    def select_all_left_page(self): self._set_all(self.dual_left_pane.tiles())
    def select_all_right_page(self): self._set_all(self.dual_right_pane.tiles())
    def _set_all(self, tiles): 
        for t in tiles or []: self.on_tile_clicked(t["h5_path"], int(t["row_idx"]), "left")