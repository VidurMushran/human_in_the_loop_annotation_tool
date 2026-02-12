from __future__ import annotations
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QMessageBox
from app.metrics.eval import binary_metrics
from app.ui.widgets.matplotlib_canvas import MplCanvas
from app.data.h5io import read_features_columns, read_images_by_indices
from app.imaging.render import channels_to_rgb8bit
from app.ui.widgets.gallery import GalleryWidget

class MetricsTab(QWidget):
    def __init__(self, cfg, annotate_tab):
        super().__init__()
        self.cfg = cfg
        self.annotate_tab = annotate_tab
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self.score_col = QLineEdit(self.cfg.default_score_col)
        self.btn_eval = QPushButton("Compute metrics on selected files")
        self.btn_eval.clicked.connect(self.compute)

        top.addWidget(QLabel("Score column:"))
        top.addWidget(self.score_col)
        top.addWidget(self.btn_eval)
        lay.addLayout(top)

        plots = QHBoxLayout()
        self.cm_canvas = MplCanvas()
        self.roc_canvas = MplCanvas()
        plots.addWidget(self.cm_canvas, 1)
        plots.addWidget(self.roc_canvas, 1)
        lay.addLayout(plots, 1)

        galrow = QHBoxLayout()
        self.g_tp = GalleryWidget()
        self.g_tn = GalleryWidget()
        self.g_fp = GalleryWidget()
        self.g_fn = GalleryWidget()
        galrow.addWidget(self.g_tp, 1)
        galrow.addWidget(self.g_tn, 1)
        galrow.addWidget(self.g_fp, 1)
        galrow.addWidget(self.g_fn, 1)
        lay.addLayout(galrow, 2)

    def compute(self):
        paths = self.annotate_tab.selected_paths
        if not paths:
            QMessageBox.information(self, "No files", "Select files in Annotate tab.")
            return
        label_col = self.annotate_tab.label_col
        score_col = self.score_col.text().strip()
        if not score_col:
            return

        y_true = []
        y_score = []
        refs = []  # (path, row_idx)
        for fp in paths:
            df = read_features_columns(fp, [label_col, score_col], features_key=self.cfg.features_key)
            s = df[label_col].astype(str).str.lower()
            mask = s.str.contains("junk") | s.str.contains("cell")
            if mask.sum() == 0:
                continue
            yt = np.where(s[mask].str.contains("junk"), 1, 0).astype(int)
            ys = df.loc[mask, score_col].astype(float).to_numpy()
            idxs = np.where(mask.to_numpy())[0]
            y_true.append(yt)
            y_score.append(ys)
            refs.extend([(fp, int(i)) for i in idxs])

        if not y_true:
            QMessageBox.warning(self, "No labeled rows", f"No rows with '{label_col}' containing 'junk' or 'cell'.")
            return

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)

        cm, (fpr, tpr, roc_auc), y_pred = binary_metrics(y_true, y_score, thr=0.5)

        # Confusion matrix plot
        ax = self.cm_canvas.ax
        ax.clear()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix @0.5")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["pred cell(0)", "pred junk(1)"])
        ax.set_yticklabels(["true cell(0)", "true junk(1)"])
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center")
        self.cm_canvas.draw()

        # ROC
        ax = self.roc_canvas.ax
        ax.clear()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0,1],[0,1], linestyle="--")
        ax.set_title("ROC")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend()
        self.roc_canvas.draw()

        # galleries (sample up to 80 each)
        tp = np.where((y_true==1) & (y_pred==1))[0]
        tn = np.where((y_true==0) & (y_pred==0))[0]
        fp = np.where((y_true==0) & (y_pred==1))[0]
        fn = np.where((y_true==1) & (y_pred==0))[0]

        self._fill_gallery(self.g_tp, refs, tp, "TP")
        self._fill_gallery(self.g_tn, refs, tn, "TN")
        self._fill_gallery(self.g_fp, refs, fp, "FP")
        self._fill_gallery(self.g_fn, refs, fn, "FN")

    def _fill_gallery(self, gal: GalleryWidget, refs, idxs, title):
        if len(idxs) == 0:
            gal.set_tiles([])
            return
        rng = np.random.default_rng(0)
        picks = rng.choice(idxs, size=min(80, len(idxs)), replace=False)

        tiles = []
        # group reads per file
        by_file = {}
        for k in picks:
            fp, ridx = refs[int(k)]
            by_file.setdefault(fp, []).append(ridx)

        rgb_cache = {}
        for fp, row_idxs in by_file.items():
            imgs = read_images_by_indices(fp, np.asarray(row_idxs), image_key=self.cfg.image_key)
            rgb_cache[fp] = dict(zip(row_idxs, [channels_to_rgb8bit(im) for im in imgs]))

        for k in picks:
            fp, ridx = refs[int(k)]
            tiles.append({
                "h5_path": fp,
                "row_idx": ridx,
                "rgb": rgb_cache[fp][ridx],
                "label": title,
                "tooltip": f"{title}\n{fp}\nidx={ridx}"
            })

        gal.set_layout(n_cols=6, tile_px=84)
        gal.set_tiles(tiles)
