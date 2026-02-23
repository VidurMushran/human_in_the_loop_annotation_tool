from __future__ import annotations
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, 
    QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QComboBox, QSplitter
)
from PyQt5.QtCore import Qt

from app.metrics.eval import binary_metrics
from app.ui.widgets.matplotlib_canvas import MplCanvas
from app.data.h5io import read_features_columns, read_images_by_indices
from app.imaging.render import channels_to_rgb8bit
from app.ui.widgets.gallery import GalleryWidget
from app.utils.qt_threading import run_in_thread

# Import new refactored helpers
from app.utils.model_helpers import load_model_from_checkpoint
from app.experiments.dashboard import parse_runs_directory
from app.training.jobs import run_scoring_job

def _basename(p: str) -> str:
    return os.path.basename(p) if p else p

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
        
        # ==========================================
        # TOP SECTION: Sweep Dashboard & Filters
        # ==========================================
        dash_widget = QWidget()
        dash_lay = QVBoxLayout(dash_widget)
        
        filter_row = QHBoxLayout()
        self.btn_refresh_runs = QPushButton("🔄 Refresh Sweeps")
        self.btn_refresh_runs.clicked.connect(self.refresh_sweep_data)
        
        self.filter_method = QComboBox()
        self.filter_method.addItem("All Methods")
        self.filter_method.currentTextChanged.connect(self.apply_filters)
        
        self.filter_inputs = QComboBox()
        self.filter_inputs.addItem("All Inputs")
        self.filter_inputs.currentTextChanged.connect(self.apply_filters)

        filter_row.addWidget(self.btn_refresh_runs)
        filter_row.addWidget(QLabel("Filter Method:"))
        filter_row.addWidget(self.filter_method)
        filter_row.addWidget(QLabel("Filter Modality:"))
        filter_row.addWidget(self.filter_inputs)
        filter_row.addStretch()
        dash_lay.addLayout(filter_row)

        # Table & Chart
        data_row = QHBoxLayout()
        
        self.runs_table = QTableWidget()
        self.runs_table.setColumnCount(6)
        self.runs_table.setHorizontalHeaderLabels(["Run Name", "Method", "Inputs", "Architecture", "Metric (Acc/SSL)", "Time (s)"])
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.runs_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.runs_table.horizontalHeader().setSectionResizeMode(0, 1) # Stretch Name column
        
        self.sweep_canvas = MplCanvas()
        
        data_row.addWidget(self.runs_table, 2)
        data_row.addWidget(self.sweep_canvas, 1)
        dash_lay.addLayout(data_row)
        
        btn_load_selected = QPushButton("Load Model from Selected Run ⬇️")
        btn_load_selected.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        btn_load_selected.clicked.connect(self.load_selected_run)
        dash_lay.addWidget(btn_load_selected)
        
        splitter.addWidget(dash_widget)

        # ==========================================
        # BOTTOM SECTION: Deep-Dive Evaluation
        # ==========================================
        eval_widget = QWidget()
        eval_lay = QVBoxLayout(eval_widget)
        
        top = QHBoxLayout()
        self.score_col = QLineEdit(self.cfg.default_score_col)
        
        self.btn_load_ckpt = QPushButton("Load Checkpoint Manually")
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
        eval_lay.addLayout(top)

        plots = QHBoxLayout()
        self.cm_canvas = MplCanvas()
        self.roc_canvas = MplCanvas()
        plots.addWidget(self.cm_canvas, 1)
        plots.addWidget(self.roc_canvas, 1)
        eval_lay.addLayout(plots, 1)

        galrow = QHBoxLayout()
        self.g_tp = GalleryWidget()
        self.g_tn = GalleryWidget()
        self.g_fp = GalleryWidget()
        self.g_fn = GalleryWidget()
        galrow.addWidget(self.g_tp, 1)
        galrow.addWidget(self.g_tn, 1)
        galrow.addWidget(self.g_fp, 1)
        galrow.addWidget(self.g_fn, 1)
        eval_lay.addLayout(galrow, 2)
        
        self.status_lbl = QLabel("Ready")
        eval_lay.addWidget(self.status_lbl)

        splitter.addWidget(eval_widget)
        main_lay.addWidget(splitter)
        splitter.setSizes([400, 600])

    # ==========================================
    # SWEEP DASHBOARD LOGIC
    # ==========================================
    def refresh_sweep_data(self):
        runs_dir = getattr(self.cfg, "runs_dir", None) or os.path.join(self.annotate_tab.root_dir or "", "runs")
        # Use helper to parse directory
        self.all_runs = parse_runs_directory(runs_dir)

        # Update Filter Dropdowns
        methods = sorted(list(set(r["method"] for r in self.all_runs)))
        inputs = sorted(list(set(r["inputs"] for r in self.all_runs)))

        self.filter_method.blockSignals(True)
        self.filter_inputs.blockSignals(True)
        self.filter_method.clear()
        self.filter_inputs.clear()
        self.filter_method.addItem("All Methods")
        self.filter_inputs.addItem("All Inputs")
        self.filter_method.addItems(methods)
        self.filter_inputs.addItems(inputs)
        self.filter_method.blockSignals(False)
        self.filter_inputs.blockSignals(False)

        self.apply_filters()

    def apply_filters(self):
        f_method = self.filter_method.currentText()
        f_inputs = self.filter_inputs.currentText()

        filtered_runs = [
            r for r in self.all_runs 
            if (f_method == "All Methods" or r["method"] == f_method) and 
               (f_inputs == "All Inputs" or r["inputs"] == f_inputs)
        ]
        # Sort by metric descending
        filtered_runs.sort(key=lambda x: x["metric"], reverse=True)

        self.runs_table.setRowCount(len(filtered_runs))
        for i, r in enumerate(filtered_runs):
            item_name = QTableWidgetItem(r["folder_name"])
            item_name.setData(Qt.UserRole, r["folder_path"]) # Store path invisibly
            
            self.runs_table.setItem(i, 0, item_name)
            self.runs_table.setItem(i, 1, QTableWidgetItem(r["method"]))
            self.runs_table.setItem(i, 2, QTableWidgetItem(r["inputs"]))
            self.runs_table.setItem(i, 3, QTableWidgetItem(r["arch"]))
            self.runs_table.setItem(i, 4, QTableWidgetItem(f"{r['metric']:.4f}"))
            self.runs_table.setItem(i, 5, QTableWidgetItem(f"{r['time']:.1f}"))

        self.plot_sweep_comparison(filtered_runs)

    def plot_sweep_comparison(self, runs):
        ax = self.sweep_canvas.ax
        ax.clear()
        if not runs:
            self.sweep_canvas.draw()
            return
            
        # Top 10 runs for readability
        plot_runs = runs[:10]
        names = [r["folder_name"].split('_', 2)[-1][:20] for r in plot_runs] # Shorten name
        metrics = [r["metric"] for r in plot_runs]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, metrics, align='center', color='#4CAF50')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Validation Metric (Acc/SSL)')
        ax.set_title('Top Model Comparison')
        self.sweep_canvas.draw()

    def load_selected_run(self):
        selected = self.runs_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a run from the table first.")
            return
        
        folder_path = self.runs_table.item(selected[0].row(), 0).data(Qt.UserRole)
        
        # Look for checkpoint
        ckpt_path = os.path.join(folder_path, "checkpoint_src.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(folder_path, "checkpoint.pt")
            
        if not os.path.exists(ckpt_path):
            QMessageBox.warning(self, "Not Found", "No checkpoint.pt found in this run directory.")
            return
            
        self._load_model_wrapper(ckpt_path)

    # ==========================================
    # DEEP DIVE LOGIC
    # ==========================================
    def load_checkpoint_clicked(self):
        default_dir = getattr(self.cfg, "runs_dir", None) or os.path.join(self.annotate_tab.root_dir or "", "runs")
        path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", default_dir, "PyTorch Checkpoints (*.pt *.pth)")
        if path:
            self._load_model_wrapper(path)
            
    def _load_model_wrapper(self, path):
        """Uses the robust helper to load and auto-detect settings."""
        try:
            self.model, kind, ch = load_model_from_checkpoint(path)
            self.status_lbl.setText(f"Loaded {kind} ({ch}ch) from {os.path.basename(path)}")
            QMessageBox.information(self, "Success", f"Successfully loaded model!\n\nDetected Channels: {ch}\nArchitecture: {kind}")
        except Exception as e:
            self.status_lbl.setText("Failed to load checkpoint")
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

        # Use the job wrapper from jobs.py (wrapped in a closure for the thread)
        def job_wrapper(log_cb):
            run_scoring_job(self.model, paths, score_col, self.cfg, log_cb)

        run_in_thread(
            job_wrapper,
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
        
        # Aggregate data from all selected files
        for fp in paths:
            try:
                df = read_features_columns(fp, [label_col, score_col], features_key=self.cfg.features_key)
                s = df[label_col].astype(str).str.lower()
                
                # Support both string labels and numeric
                mask_junk = s.str.contains("junk") | (s == "1.0") | (s == "1")
                mask_cell = s.str.contains("cell") | (s == "0.0") | (s == "0")
                mask = mask_junk | mask_cell
                
                if mask.sum() == 0:
                    continue
                    
                yt = np.where(mask_junk[mask], 1, 0).astype(int)
                ys = df.loc[mask, score_col].astype(float).to_numpy()
                
                # Filter NaNs
                valid_score_mask = ~np.isnan(ys)
                yt = yt[valid_score_mask]
                ys = ys[valid_score_mask]
                
                idxs = np.where(mask.to_numpy())[0][valid_score_mask]
                
                y_true.append(yt)
                y_score.append(ys)
                refs.extend([(fp, int(i)) for i in idxs])
            except Exception as e:
                print(f"Skipping {fp}: {e}")

        if not y_true or len(np.concatenate(y_true)) == 0:
            QMessageBox.warning(self, "No valid rows", f"No valid rows with '{label_col}' and finite '{score_col}'.")
            return

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)

        # Calculate Metrics
        cm, (fpr, tpr, roc_auc), y_pred = binary_metrics(y_true, y_score, thr=0.5)

        # Plot Confusion Matrix
        ax = self.cm_canvas.ax
        ax.clear()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix @0.5")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["pred cell", "pred junk"])
        ax.set_yticklabels(["true cell", "true junk"])
        for (i,j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center")
        self.cm_canvas.draw()

        # Plot ROC
        ax = self.roc_canvas.ax
        ax.clear()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0,1],[0,1], linestyle="--")
        ax.set_title("ROC")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend()
        self.roc_canvas.draw()

        # Populate Galleries (TP, TN, FP, FN)
        tp = np.where((y_true==1) & (y_pred==1))[0]
        tn = np.where((y_true==0) & (y_pred==0))[0]
        fp = np.where((y_true==0) & (y_pred==1))[0]
        fn = np.where((y_true==1) & (y_pred==0))[0]

        self._fill_gallery(self.g_tp, refs, tp, "TP (Junk classified as Junk)")
        self._fill_gallery(self.g_tn, refs, tn, "TN (Cells classified as Cells)")
        self._fill_gallery(self.g_fp, refs, fp, "FP (Cells classified as Junk)")
        self._fill_gallery(self.g_fn, refs, fn, "FN (Junk classified as Cells)")

    def _fill_gallery(self, gal: GalleryWidget, refs, idxs, title):
        if len(idxs) == 0:
            gal.set_tiles([])
            return
        # Sample max 80 images for performance
        rng = np.random.default_rng(0)
        picks = rng.choice(idxs, size=min(80, len(idxs)), replace=False)

        tiles = []
        by_file = {}
        for k in picks:
            fp, ridx = refs[int(k)]
            by_file.setdefault(fp, []).append(ridx)

        # Batch read images
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
                "tooltip": f"{title}\n{os.path.basename(fp)}\nidx={ridx}"
            })

        gal.set_layout(n_cols=6, tile_h=84, tile_w=84)
        gal.set_tiles(tiles)