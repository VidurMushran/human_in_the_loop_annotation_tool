# app/ui/train_tab.py
from __future__ import annotations
import os
import itertools
import h5py
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QMessageBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QAbstractItemView, QListWidget)
from PyQt5.QtCore import Qt

from ..ui.widgets.multicheck_dropdown import MultiCheckDropdown
from ..utils.qt_threading import run_in_thread, JobQueueManager
from ..utils.model_helpers import load_model_from_checkpoint
from ..training.jobs import run_standard_job, run_pseudo_label_job, run_scoring_job
from ..ml.train import TrainConfig
from ..ml.dataset import Item

def _basename(p): return os.path.basename(p) if p else p
def _make_train_config(**kwargs): return TrainConfig(**kwargs)

class TrainTab(QWidget):
    def __init__(self, cfg, annotate_tab):
        super().__init__()
        self.cfg = cfg
        self.annotate_tab = annotate_tab
        self.model = None
        
        self.queue_manager = JobQueueManager(self)
        self.queue_manager.log_signal.connect(self._log)
        self.queue_manager.job_started.connect(self._on_job_started)
        self.queue_manager.job_finished.connect(self._on_job_finished)
        self.queue_manager.queue_empty.connect(lambda: self._log("Queue is now empty."))
        
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.model_menu = MultiCheckDropdown("Architectures")
        self.model_menu.add_option("simple_cnn", True); self.model_menu.add_option("timm:resnet18", False); self.model_menu.add_option("timm:resnet50", False)
        self.inputs_menu = MultiCheckDropdown("Input Modalities")
        self.inputs_menu.add_option("image_only", True); self.inputs_menu.add_option("image_and_mask", False); self.inputs_menu.add_option("image_and_features", False)
        self.method_menu = MultiCheckDropdown("Training Method")
        self.method_menu.add_option("supervised", True); self.method_menu.add_option("self-supervised", False); self.method_menu.add_option("pseudo-labeling", False)
        self.aug_menu = MultiCheckDropdown("Augmentations")
        self.aug_menu.add_option("hflip", True); self.aug_menu.add_option("vflip", False); self.aug_menu.add_option("rotate90", True); self.aug_menu.add_option("gaussian_blur", False)
        row.addWidget(self.model_menu); row.addWidget(self.inputs_menu); row.addWidget(self.method_menu); row.addWidget(self.aug_menu)
        lay.addLayout(row)

        rowS = QHBoxLayout()
        self.cluster_col = QLineEdit("cluster")
        self.blur_sigma = QDoubleSpinBox(); self.blur_sigma.setRange(0, 10); self.blur_sigma.setValue(1.5)
        self.test_frac = QDoubleSpinBox(); self.test_frac.setValue(0.20)
        self.val_frac = QDoubleSpinBox(); self.val_frac.setValue(0.20)
        self.pl_iters = QDoubleSpinBox(); self.pl_iters.setValue(3)
        self.pl_thresh = QDoubleSpinBox(); self.pl_thresh.setValue(0.90)
        rowS.addWidget(QLabel("Max blur:")); rowS.addWidget(self.blur_sigma)
        rowS.addWidget(QLabel("Cluster col:")); rowS.addWidget(self.cluster_col)
        rowS.addWidget(QLabel("Test frac:")); rowS.addWidget(self.test_frac)
        rowS.addWidget(QLabel("Val frac:")); rowS.addWidget(self.val_frac)
        rowS.addWidget(QLabel("PL Iters:")); rowS.addWidget(self.pl_iters)
        rowS.addWidget(QLabel("PL Thresh:")); rowS.addWidget(self.pl_thresh)
        lay.addLayout(rowS)

        self.file_table = QTableWidget(); self.file_table.setColumnCount(3)
        self.file_table.setHorizontalHeaderLabels(["Use?", "HDF5", "Label Mode"])
        self.btn_refresh = QPushButton("Refresh"); self.btn_refresh.clicked.connect(self._refresh_file_table)
        lay.addWidget(QLabel("Training Files:")); lay.addWidget(self.file_table, 1); lay.addWidget(self.btn_refresh)

        q_row = QHBoxLayout()
        self.btn_sweep = QPushButton("Generate Sweep"); self.btn_sweep.clicked.connect(self.queue_sweep_clicked)
        self.queue_list = QListWidget(); self.queue_list.setMaximumHeight(100)
        q_row.addWidget(self.btn_sweep); 
        lay.addLayout(q_row); lay.addWidget(QLabel("Queue:")); lay.addWidget(self.queue_list)

        row2 = QHBoxLayout()
        self.score_col = QLineEdit(f"model_score_{time.strftime('%Y%m%d_%H%M%S')}")
        self.btn_load = QPushButton("Load Checkpoint"); self.btn_load.clicked.connect(self.load_checkpoint_clicked)
        self.btn_score = QPushButton("Score Selected"); self.btn_score.clicked.connect(self.score_clicked)
        row2.addWidget(QLabel("Score col:")); row2.addWidget(self.score_col); row2.addWidget(self.btn_load); row2.addWidget(self.btn_score)
        lay.addLayout(row2)

        self.log = QTextEdit(); self.log.setReadOnly(True); lay.addWidget(self.log, 2)

    def _log(self, s): self.log.append(str(s)); self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
    def _on_job_started(self, n): 
        if i := self.queue_list.findItems(n, Qt.MatchContains): i[0].setText(f"🔄 {n}"); i[0].setBackground(Qt.yellow)
    def _on_job_finished(self, n, s): 
        if i := self.queue_list.findItems(n, Qt.MatchContains): i[0].setText(f"{'✅' if s else '❌'} {n}"); i[0].setBackground(Qt.green if s else Qt.red)

    def _refresh_file_table(self):
        if not self.annotate_tab.root_dir: return
        from app.data.discover import discover_hdf5s
        files = [fp for _, fp in discover_hdf5s(self.annotate_tab.root_dir)]
        self.file_table.setRowCount(len(files))
        for r, fp in enumerate(files):
            self.file_table.setItem(r, 0, QTableWidgetItem("")); self.file_table.item(r,0).setCheckState(0)
            self.file_table.setItem(r, 1, QTableWidgetItem(fp))
            cb = QComboBox(); cb.addItems(["use_label_col", "fixed_0", "fixed_1", "unlabeled (all rows)"])
            self.file_table.setCellWidget(r, 2, cb)

    def _get_supervised_file_specs(self):
        specs = []
        for r in range(self.file_table.rowCount()):
            if self.file_table.item(r, 0).checkState() == 2:
                specs.append((self.file_table.item(r, 1).text(), self.file_table.cellWidget(r, 2).currentText()))
        return specs

    def _items_from_file(self, fp, mode, label_col, cluster_col):
        try: df = pd.read_hdf(fp, key=self.cfg.features_key)
        except: return []
        if mode == "fixed_0": y = np.zeros(len(df), int)
        elif mode == "fixed_1": y = np.ones(len(df), int)
        else:
            if label_col not in df.columns: return []
            s = df[label_col].astype(str).str.lower()
            ji = np.where(s.str.contains("junk"))[0]; ci = np.where(s.str.contains("cell"))[0]
            idxs = np.concatenate([ji, ci]); y = np.concatenate([np.ones(len(ji), int), np.zeros(len(ci), int)])
            df = df.iloc[idxs]
        
        cl = np.full(len(df), -1, int)
        if cluster_col in df.columns:
            try: cl = df[cluster_col].fillna(-1).astype(int).to_numpy()
            except: cl = np.array([hash(x) % 10**9 for x in df[cluster_col].fillna(-1)], int)
        return [Item(fp, i, yy, c) for i, yy, c in zip(df.index, y, cl)]

    def load_checkpoint_clicked(self):
        from PyQt5.QtWidgets import QFileDialog
        d = self.annotate_tab.root_dir or ""
        p, _ = QFileDialog.getOpenFileName(self, "Load Checkpoint", d, "*.pt *.pth")
        if not p: return
        try:
            self.model, kind, ch = load_model_from_checkpoint(p)
            self._log(f"Loaded {kind} ({ch}ch) from {p}"); QMessageBox.information(self, "Loaded", f"Model ready.")
        except Exception as e: self._log(f"Error: {e}"); QMessageBox.warning(self, "Error", str(e))

    def queue_sweep_clicked(self):
        if not self.annotate_tab.root_dir: return QMessageBox.warning(self, "No root", "Select root dir first.")
        specs = self._get_supervised_file_specs()
        if not specs: return QMessageBox.warning(self, "No files", "Select files.")
        
        # Auto-detect base channels
        base_ch = 3
        try:
            with h5py.File(specs[0][0], 'r') as f:
                if len(f[self.cfg.image_key].shape) == 4: base_ch = f[self.cfg.image_key].shape[-1]
        except: pass

        models = self.model_menu.selected(); inputs = self.inputs_menu.selected(); methods = self.method_menu.selected()
        if not (models and inputs and methods): return QMessageBox.warning(self, "Incomplete", "Select options.")

        # Gather items
        lbl_col = self.annotate_tab.label_col; cl_col = self.cluster_col.text().strip()
        labeled, unlabeled = [], []
        for fp, mode in specs:
            if mode == "unlabeled (all rows)":
                try: 
                    df = pd.read_hdf(fp, key=self.cfg.features_key)
                    unlabeled.extend([Item(fp, i, -1, -1) for i in df.index])
                except: pass
            else: labeled.extend(self._items_from_file(fp, mode, lbl_col, cl_col))

        combs = list(itertools.product(models, inputs, methods))
        self._log(f"Generated {len(combs)} sweep jobs.")
        runs_root = getattr(self.cfg, "runs_dir", None) or os.path.join(self.annotate_tab.root_dir, "runs")

        for arch, inp, meth in combs:
            kind, tname = ("timm_frozen", arch.split(":")[1]) if "timm" in arch else ("simple_cnn", "resnet18")
            job_name = f"{arch} | {inp} | {meth}"
            self.queue_list.addItem(f"⏳ {job_name}")
            
            cfg = _make_train_config(model_kind=kind, timm_name=tname, inputs_mode=inp, training_method=meth,
                                     epochs=5, batch_size=256, device="cuda", image_key=self.cfg.image_key,
                                     aug_flags=tuple(self.aug_menu.selected()), max_blur_sigma=self.blur_sigma.value())
            
            ch = base_ch + (1 if inp == "image_and_mask" else 0)
            
            if meth == "pseudo-labeling":
                iters = int(self.pl_iters.value()); th = self.pl_thresh.value()
                self.queue_manager.add_job(job_name, lambda cb, c=cfg, n=job_name, l=labeled, u=unlabeled, tn=tname, ch=ch, r=runs_root, i=iters, t=th: self._set_model(run_pseudo_label_job(c,n,l,u,tn,ch,r,i,t,cb)))
            else:
                # Random split
                rng = np.random.default_rng(0); idx = rng.permutation(len(labeled)); n_val = int(len(labeled)*self.val_frac.value())
                val_items = [labeled[i] for i in idx[:n_val]]; train_items = [labeled[i] for i in idx[n_val:]]
                self.queue_manager.add_job(job_name, lambda cb, c=cfg, n=job_name, tr=train_items, va=val_items, tn=tname, ch=ch, r=runs_root: self._set_model(run_standard_job(c,n,tr,va,tn,ch,r,cb)))

    def _set_model(self, m): self.model = m
    def score_clicked(self):
        if not self.model: return QMessageBox.warning(self, "No model", "Load/Train model first.")
        paths = self.annotate_tab.selected_paths
        if not paths: return QMessageBox.warning(self, "No files", "Select files.")
        run_in_thread(lambda cb: run_scoring_job(self.model, paths, self.score_col.text(), self.cfg, cb), 
                      parent=self, on_log=self._log, on_done=lambda: self._log("Scoring done."))