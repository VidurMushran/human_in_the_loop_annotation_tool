# app/ui/metrics_tab.py
from __future__ import annotations
import os
import torch
import numpy as np
import pandas as pd
import h5py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, 
    QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QComboBox, QSplitter, QGridLayout
)
from PyQt5.QtCore import Qt

from app.metrics.eval import binary_metrics
from app.ui.widgets.matplotlib_canvas import MplCanvas
from app.data.h5io import read_features_columns, read_images_by_indices
from app.imaging.render import channels_to_rgb8bit
from app.ui.widgets.gallery import GalleryWidget
from app.utils.qt_threading import run_in_thread
from app.utils.model_helpers import load_model_from_checkpoint
from app.experiments.dashboard import parse_runs_directory
from app.training.jobs import run_scoring_job

try: import umap
except: umap = None

class MetricsTab(QWidget):
    def __init__(self, cfg, annotate_tab):
        super().__init__()
        self.cfg = cfg
        self.annotate_tab = annotate_tab
        self.model = None
        self.all_runs = [] 
        self._build()

    def _build(self):
        main_lay = QVBoxLayout(self)
        splitter = QSplitter(Qt.Vertical)
        
        # --- Dashboard ---
        dash = QWidget(); dl = QVBoxLayout(dash)
        fr = QHBoxLayout()
        self.btn_refresh = QPushButton("🔄 Refresh Sweeps"); self.btn_refresh.clicked.connect(self.refresh_sweep_data)
        self.f_meth = QComboBox(); self.f_meth.currentTextChanged.connect(self.apply_filters)
        self.f_inp = QComboBox(); self.f_inp.currentTextChanged.connect(self.apply_filters)
        fr.addWidget(self.btn_refresh); fr.addWidget(QLabel("Method:")); fr.addWidget(self.f_meth); fr.addWidget(QLabel("Input:")); fr.addWidget(self.f_inp); fr.addStretch()
        dl.addLayout(fr)

        dr = QHBoxLayout()
        self.table = QTableWidget(); self.table.setColumnCount(6); self.table.setHorizontalHeaderLabels(["Run","Method","Input","Arch","Metric","Time"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows); self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, 1)
        self.chart = MplCanvas()
        dr.addWidget(self.table, 2); dr.addWidget(self.chart, 1)
        dl.addLayout(dr)
        
        btn_load = QPushButton("Load Model from Selected Run ⬇️"); btn_load.clicked.connect(self.load_selected_run)
        dl.addWidget(btn_load)
        splitter.addWidget(dash)

        # --- Deep Dive ---
        ev = QWidget(); el = QVBoxLayout(ev)
        tr = QHBoxLayout()
        self.score_col = QLineEdit(self.cfg.default_score_col)
        self.btn_man_load = QPushButton("Load Checkpoint"); self.btn_man_load.clicked.connect(self.load_checkpoint_clicked)
        self.btn_score = QPushButton("Score Selected"); self.btn_score.clicked.connect(self.score_clicked)
        self.btn_eval = QPushButton("Compute Metrics"); self.btn_eval.clicked.connect(self.compute)
        self.btn_umap = QPushButton("Generate UMAP"); self.btn_umap.clicked.connect(self.run_umap)
        
        tr.addWidget(QLabel("Score col:")); tr.addWidget(self.score_col); tr.addWidget(self.btn_man_load); tr.addWidget(self.btn_score); tr.addWidget(self.btn_eval); tr.addWidget(self.btn_umap)
        el.addLayout(tr)

        pr = QHBoxLayout(); self.cm_plot = MplCanvas(); self.roc_plot = MplCanvas()
        pr.addWidget(self.cm_plot); pr.addWidget(self.roc_plot); el.addLayout(pr)
        
        # Galleries (4 Panes with Download)
        gr = QGridLayout()
        self.gal_tp = GalleryWidget("TP"); gr.addWidget(self.gal_tp, 0, 0)
        self.gal_tn = GalleryWidget("TN"); gr.addWidget(self.gal_tn, 0, 1)
        self.gal_fp = GalleryWidget("FP"); gr.addWidget(self.gal_fp, 1, 0)
        self.gal_fn = GalleryWidget("FN"); gr.addWidget(self.gal_fn, 1, 1)
        el.addLayout(gr, 2)
        
        dl_row = QHBoxLayout()
        btn_dl = QPushButton("Download Galleries (All Images)"); btn_dl.clicked.connect(self.download_galleries)
        dl_row.addStretch(); dl_row.addWidget(btn_dl); dl_row.addStretch()
        el.addLayout(dl_row)
        
        self.lbl_status = QLabel("Ready"); el.addWidget(self.lbl_status)
        splitter.addWidget(ev)
        main_lay.addWidget(splitter); splitter.setSizes([400, 600])

        # Cache for galleries to allow download
        self.gallery_cache = {"TP":[], "TN":[], "FP":[], "FN":[]}

    def refresh_sweep_data(self):
        d = getattr(self.cfg, "runs_dir", None) or os.path.join(self.annotate_tab.root_dir or "", "runs")
        self.all_runs = parse_runs_directory(d)
        methods = sorted(list(set(r["method"] for r in self.all_runs)))
        inputs = sorted(list(set(r["inputs"] for r in self.all_runs)))
        self.f_meth.blockSignals(True); self.f_inp.blockSignals(True)
        self.f_meth.clear(); self.f_meth.addItem("All"); self.f_meth.addItems(methods)
        self.f_inp.clear(); self.f_inp.addItem("All"); self.f_inp.addItems(inputs)
        self.f_meth.blockSignals(False); self.f_inp.blockSignals(False)
        self.apply_filters()

    def apply_filters(self):
        m = self.f_meth.currentText(); i = self.f_inp.currentText()
        runs = [r for r in self.all_runs if (m=="All" or r["method"]==m) and (i=="All" or r["inputs"]==i)]
        runs.sort(key=lambda x: x["metric"], reverse=True)
        self.table.setRowCount(len(runs))
        for idx, r in enumerate(runs):
            it = QTableWidgetItem(r["folder_name"]); it.setData(Qt.UserRole, r["folder_path"])
            self.table.setItem(idx, 0, it); self.table.setItem(idx, 1, QTableWidgetItem(r["method"]))
            self.table.setItem(idx, 2, QTableWidgetItem(r["inputs"])); self.table.setItem(idx, 3, QTableWidgetItem(r["arch"]))
            self.table.setItem(idx, 4, QTableWidgetItem(f"{r['metric']:.4f}")); self.table.setItem(idx, 5, QTableWidgetItem(f"{r['time']:.1f}"))
        self._plot_bars(runs[:10])

    def _plot_bars(self, runs):
        ax = self.chart.ax; ax.clear()
        if not runs: self.chart.draw(); return
        names = [r["folder_name"][-20:] for r in runs]; vals = [r["metric"] for r in runs]
        y = np.arange(len(names)); ax.barh(y, vals, color='#4CAF50')
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis(); self.chart.draw()

    def load_selected_run(self):
        sel = self.table.selectedItems()
        if not sel: return
        p = sel[0].data(Qt.UserRole)
        ckpt = os.path.join(p, "checkpoint_src.pt")
        if not os.path.exists(ckpt): ckpt = os.path.join(p, "checkpoint.pt")
        self._load_model(ckpt)

    def load_checkpoint_clicked(self):
        d = getattr(self.cfg, "runs_dir", None) or ""
        p, _ = QFileDialog.getOpenFileName(self, "Load", d, "*.pt")
        if p: self._load_model(p)

    def _load_model(self, path):
        try:
            self.model, kind, ch = load_model_from_checkpoint(path)
            self.lbl_status.setText(f"Loaded {kind} ({ch}ch)"); QMessageBox.information(self, "Loaded", "Model Ready")
        except Exception as e: self.lbl_status.setText("Error"); QMessageBox.warning(self, "Error", str(e))

    def score_clicked(self):
        if not self.model: return QMessageBox.warning(self, "No model", "Load model.")
        paths = self.annotate_tab.selected_paths
        if not paths: return QMessageBox.warning(self, "No files", "Select files.")
        run_in_thread(lambda cb: run_scoring_job(self.model, paths, self.score_col.text(), self.cfg, cb), 
                      parent=self, on_log=lambda s: self.lbl_status.setText(s), on_done=lambda: self.lbl_status.setText("Done."))

    def compute(self):
        paths = self.annotate_tab.selected_paths
        if not paths: return QMessageBox.warning(self, "No files", "Select files.")
        lc = self.annotate_tab.label_col; sc = self.score_col.text()
        y_true, y_score, refs = [], [], []
        
        for fp in paths:
            try:
                df = read_features_columns(fp, [lc, sc], features_key=self.cfg.features_key)
                s = df[lc].astype(str).str.lower()
                mj = s.str.contains("junk") | (s=="1"); mc = s.str.contains("cell") | (s=="0")
                mask = mj|mc
                if mask.sum()==0: 
                    continue
                yt = np.where(mj[mask], 1, 0).astype(int)
                ys = df.loc[mask, sc].astype(float).to_numpy()
                valid = ~np.isnan(ys); yt, ys = yt[valid], ys[valid]
                idxs = np.where(mask.to_numpy())[0][valid]
                y_true.append(yt); y_score.append(ys); refs.extend([(fp, int(i)) for i in idxs])
            except: pass

        if not y_true: return QMessageBox.warning(self, "No data", "No labeled data found.")
        y_true = np.concatenate(y_true); y_score = np.concatenate(y_score)
        
        cm, (fpr, tpr, auc), y_pred = binary_metrics(y_true, y_score, thr=0.5)
        
        ax = self.cm_plot.ax; ax.clear(); ax.imshow(cm); ax.set_title("Confusion Matrix")
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,int(v),ha='center',va='center')
        self.cm_plot.draw()
        
        ax = self.roc_plot.ax; ax.clear(); ax.plot(fpr, tpr, label=f"AUC={auc:.3f}"); ax.legend(); self.roc_plot.draw()
        
        # Populate Galleries
        self.gallery_cache["TP"] = [refs[i] for i in np.where((y_true==1)&(y_pred==1))[0]]
        self.gallery_cache["TN"] = [refs[i] for i in np.where((y_true==0)&(y_pred==0))[0]]
        self.gallery_cache["FP"] = [refs[i] for i in np.where((y_true==0)&(y_pred==1))[0]]
        self.gallery_cache["FN"] = [refs[i] for i in np.where((y_true==1)&(y_pred==0))[0]]
        
        self._fill_gallery(self.gal_tp, self.gallery_cache["TP"], "TP")
        self._fill_gallery(self.gal_tn, self.gallery_cache["TN"], "TN")
        self._fill_gallery(self.gal_fp, self.gallery_cache["FP"], "FP")
        self._fill_gallery(self.gal_fn, self.gallery_cache["FN"], "FN")

    def _fill_gallery(self, gal, items, title):
        if not items: gal.set_tiles([]); return
        rng = np.random.default_rng(0); picks = rng.choice(len(items), size=min(80, len(items)), replace=False)
        tiles = []
        # Group by file for batch read
        by_file = {}
        for idx in picks:
            fp, ridx = items[idx]
            by_file.setdefault(fp, []).append(ridx)
        
        rgb_cache = {}
        for fp, row_idxs in by_file.items():
            imgs = read_images_by_indices(fp, np.array(row_idxs), image_key=self.cfg.image_key)
            rgb_cache[fp] = dict(zip(row_idxs, [channels_to_rgb8bit(im) for im in imgs]))
            
        for idx in picks:
            fp, ridx = items[idx]
            tiles.append({"h5_path":fp, "row_idx":ridx, "rgb":rgb_cache[fp][ridx], "label":title, "tooltip":f"{title}\n{os.path.basename(fp)}\n{ridx}"})
        gal.set_tiles(tiles)

    def download_galleries(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Dir")
        if not d: return
        self.lbl_status.setText("Saving images...")
        import cv2
        count = 0
        for cat, items in self.gallery_cache.items():
            cat_dir = os.path.join(d, cat)
            os.makedirs(cat_dir, exist_ok=True)
            # Group by file for efficient read
            by_file = {}
            for fp, ridx in items: by_file.setdefault(fp, []).append(ridx)
            
            for fp, rows in by_file.items():
                imgs = read_images_by_indices(fp, np.array(rows), image_key=self.cfg.image_key)
                for ridx, im in zip(rows, imgs):
                    rgb = channels_to_rgb8bit(im)
                    # Convert RGB to BGR for OpenCV
                    cv2.imwrite(os.path.join(cat_dir, f"{os.path.basename(fp)}_{ridx}.png"), rgb[...,::-1])
                    count += 1
        self.lbl_status.setText(f"Saved {count} images.")
        QMessageBox.information(self, "Done", f"Saved {count} images to {d}")

    def run_umap(self):
        if not umap: return QMessageBox.warning(self, "Error", "umap-learn not installed.")
        if not self.model: return QMessageBox.warning(self, "Error", "Load model first.")
        
        # 1. Gather Data (Sample 2000 items)
        paths = self.annotate_tab.selected_paths
        all_refs = []
        for fp in paths:
            try:
                with h5py.File(fp,'r') as f: n=f[self.cfg.image_key].shape[0]
                all_refs.extend([(fp, i) for i in range(n)])
            except: pass
        
        if not all_refs: return
        rng = np.random.default_rng(0)
        picks_idx = rng.choice(len(all_refs), size=min(2000, len(all_refs)), replace=False)
        picks = [all_refs[i] for i in picks_idx]
        
        # 2. Extract Features
        feats = []
        self.model.eval()
        self.model.to("cuda")
        
        # Batch extraction logic needed here...
        # Simplified:
        by_file = {}
        for fp, r in picks: by_file.setdefault(fp, []).append(r)
        
        with torch.no_grad():
            for fp, rows in by_file.items():
                imgs = read_images_by_indices(fp, np.array(rows), image_key=self.cfg.image_key)
                xb = torch.from_numpy(imgs).permute(0,3,1,2).float().cuda()
                # Assuming model returns embeddings directly or via hook
                # For now using forward (which returns embeddings in new model structure)
                emb = self.model(xb).cpu().numpy()
                feats.append(emb)
        
        feats = np.concatenate(feats)
        
        # 3. UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(feats)
        
        # 4. Plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        plt.scatter(embedding[:,0], embedding[:,1], s=5)
        plt.title("UMAP of Embeddings")
        plt.show()