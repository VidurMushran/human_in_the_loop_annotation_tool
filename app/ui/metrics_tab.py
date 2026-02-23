from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QMessageBox, QFileDialog
from app.metrics.eval import binary_metrics
from app.ui.widgets.matplotlib_canvas import MplCanvas
from app.data.h5io import read_features_columns, read_images_by_indices
from app.imaging.render import channels_to_rgb8bit
from app.ui.widgets.gallery import GalleryWidget
from app.ml.models import SimpleCNN, make_timm_frozen_linear
from app.ml.score import score_h5_file
from app.utils.qt_threading import run_in_thread

def _basename(p: str) -> str:
    return os.path.basename(p) if p else p

class MetricsTab(QWidget):
    def __init__(self, cfg, annotate_tab):
        super().__init__()
        self.cfg = cfg
        self.annotate_tab = annotate_tab
        self.model = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self.score_col = QLineEdit(self.cfg.default_score_col)
        
        self.btn_load_ckpt = QPushButton("Load Model Checkpoint")
        self.btn_load_ckpt.clicked.connect(self.load_checkpoint_clicked)
        
        self.btn_score = QPushButton("Score selected HDF5s")
        self.btn_score.clicked.connect(self.score_clicked)

        self.btn_eval = QPushButton("Compute metrics on selected files")
        self.btn_eval.clicked.connect(self.compute)

        top.addWidget(QLabel("Score col:"))
        top.addWidget(self.score_col)
        top.addWidget(self.btn_load_ckpt)
        top.addWidget(self.btn_score)
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
        
        self.status_lbl = QLabel("Ready")
        lay.addWidget(self.status_lbl)

    def load_checkpoint_clicked(self):
        default_dir = "/mnt/deepstore/Vidur/Junk_Classification/junk_gui_app/runs"
        if not os.path.exists(default_dir):
            default_dir = str(self.annotate_tab.root_dir or "")

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", default_dir, "PyTorch Checkpoints (*.pt *.pth)"
        )
        if not path:
            return

        try:
            ckpt = torch.load(path, map_location="cpu")
            t_cfg = ckpt.get("train_config", {})
            kind = t_cfg.get("model_kind", "simple_cnn")
            timm_name = t_cfg.get("timm_name", "resnet18")

            if kind == "simple_cnn":
                model = SimpleCNN(in_ch=4, n_classes=2)
            else:
                model = make_timm_frozen_linear(timm_name, in_chans=4, n_classes=2)

            model.load_state_dict(ckpt["model_state_dict"])
            self.model = model
            
            self.status_lbl.setText(f"Loaded {kind} model from {os.path.basename(path)}")
            QMessageBox.information(self, "Success", f"Model loaded from:\n{path}")
        except Exception as e:
            self.status_lbl.setText(f"Failed to load checkpoint")
            QMessageBox.warning(self, "Load Error", f"Failed to load checkpoint:\n{e}")

    def score_clicked(self):
        if self.model is None:
            QMessageBox.information(self, "No model", "Load a model checkpoint first.")
            return

        paths = self.annotate_tab.selected_paths
        if not paths:
            QMessageBox.information(self, "No files", "Select at least one HDF5 in the Annotate tab.")
            return

        score_col = self.score_col.text().strip()
        if not score_col:
            return

        self.status_lbl.setText(f"Scoring {len(paths)} file(s) into column: {score_col}...")
        self.btn_score.setEnabled(False)

        def _score_job(log_cb):
            for fp in paths:
                try:
                    log_cb(f"Scoring {_basename(fp)} ...")
                    score_h5_file(
                        self.model, fp,
                        score_col=score_col,
                        image_key=self.cfg.image_key,
                        features_key=self.cfg.features_key,
                        device="cuda",
                        batch_size=256,
                        target_hw=75,
                        log_cb=log_cb,
                    )
                except Exception as e:
                    log_cb(f"Failed to score {_basename(fp)}: {e}")

        run_in_thread(
            _score_job,
            parent=self,
            on_log=lambda s: self.status_lbl.setText(s),
            on_error=lambda e: QMessageBox.warning(self, "Scoring Error", str(e)),
            on_done=lambda: [self.status_lbl.setText("Scoring finished. You can now compute metrics."), self.btn_score.setEnabled(True)],
        )

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
            
            # Support both string labels ("junk"/"cell") and numeric representations (1.0/0.0)
            mask_junk = s.str.contains("junk") | (s == "1.0") | (s == "1")
            mask_cell = s.str.contains("cell") | (s == "0.0") | (s == "0")
            mask = mask_junk | mask_cell
            
            if mask.sum() == 0:
                continue
                
            yt = np.where(mask_junk[mask], 1, 0).astype(int)
            ys = df.loc[mask, score_col].astype(float).to_numpy()
            
            # Filter out any NaNs that might have been loaded before metrics parsing
            valid_score_mask = ~np.isnan(ys)
            yt = yt[valid_score_mask]
            ys = ys[valid_score_mask]
            
            idxs = np.where(mask.to_numpy())[0][valid_score_mask]
            
            y_true.append(yt)
            y_score.append(ys)
            refs.extend([(fp, int(i)) for i in idxs])

        if not y_true or len(np.concatenate(y_true)) == 0:
            QMessageBox.warning(self, "No valid rows", f"No valid rows with '{label_col}' (junk/cell/1/0) and finite '{score_col}'.")
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

        gal.set_layout(n_cols=6, tile_h=84, tile_w=84)
        gal.set_tiles(tiles)