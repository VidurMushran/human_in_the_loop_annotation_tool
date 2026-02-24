# app/ui/train_tab.py
from __future__ import annotations
import os
import itertools
import h5py
import pickle
import subprocess
import time
import json
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QMessageBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QAbstractItemView, QListWidget, QFileDialog)
from PyQt5.QtCore import Qt, QSettings
import pandas as pd

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
        self.settings = QSettings("VidurLab", "JunkAnnotator")
        
        self.queue_manager = JobQueueManager(self)
        self.queue_manager.log_signal.connect(self._log)
        self.queue_manager.job_started.connect(self._on_job_started)
        self.queue_manager.job_finished.connect(self._on_job_finished)
        self.queue_manager.queue_empty.connect(lambda: self._log("Queue finished."))
        self._build()
        self._load_ui_state()

    def _build(self):
        lay = QVBoxLayout(self)
        
        # Row 1: Sweep Config
        row = QHBoxLayout()
        self.model_menu = MultiCheckDropdown("Architectures")
        self.model_menu.add_option("simple_cnn", True); self.model_menu.add_option("timm:resnet18", False)
        self.inputs_menu = MultiCheckDropdown("Input Modalities")
        self.inputs_menu.add_option("image_only", True); self.inputs_menu.add_option("image_and_mask", False)
        self.method_menu = MultiCheckDropdown("Training Method")
        self.method_menu.add_option("supervised", True); self.method_menu.add_option("self-supervised", False); self.method_menu.add_option("pseudo-labeling", False)
        self.aug_menu = MultiCheckDropdown("Augmentations")
        self.aug_menu.add_option("hflip", True); self.aug_menu.add_option("vflip", False); self.aug_menu.add_option("rotate90", True)
        
        row.addWidget(self.model_menu); row.addWidget(self.inputs_menu); row.addWidget(self.method_menu); row.addWidget(self.aug_menu)
        lay.addLayout(row)

        # Row 2: Hyperparams & Resume
        rowS = QHBoxLayout()
        self.cluster_col = QLineEdit("cluster")
        self.epochs = QDoubleSpinBox(); self.epochs.setDecimals(0); self.epochs.setValue(5); self.epochs.setRange(1, 1000)
        self.blur_sigma = QDoubleSpinBox(); self.blur_sigma.setValue(1.5)
        self.resume_path = QLineEdit(); self.resume_path.setPlaceholderText("Resume Checkpoint Path (Optional)")
        self.btn_browse_resume = QPushButton("..."); self.btn_browse_resume.clicked.connect(self.browse_resume)
        
        rowS.addWidget(QLabel("Epochs:")); rowS.addWidget(self.epochs)
        rowS.addWidget(QLabel("Cluster col:")); rowS.addWidget(self.cluster_col)
        rowS.addWidget(QLabel("Resume:")); rowS.addWidget(self.resume_path); rowS.addWidget(self.btn_browse_resume)
        lay.addLayout(rowS)

        # Row 3: PL Settings & Split
        rowP = QHBoxLayout()
        self.pl_iters = QDoubleSpinBox(); self.pl_iters.setValue(3)
        self.pl_thresh = QDoubleSpinBox(); self.pl_thresh.setValue(0.90)
        self.test_frac = QDoubleSpinBox(); self.test_frac.setValue(0.20)
        rowP.addWidget(QLabel("PL Iters:")); rowP.addWidget(self.pl_iters)
        rowP.addWidget(QLabel("PL Thresh:")); rowP.addWidget(self.pl_thresh)
        rowP.addWidget(QLabel("Test Frac:")); rowP.addWidget(self.test_frac)
        lay.addLayout(rowP)

        # File Table
        self.file_table = QTableWidget(); self.file_table.setColumnCount(3)
        self.file_table.setHorizontalHeaderLabels(["Use?", "HDF5", "Label Mode"])
        self.btn_refresh = QPushButton("Refresh"); self.btn_refresh.clicked.connect(self._refresh_file_table)
        lay.addWidget(QLabel("Data Selection:")); lay.addWidget(self.file_table, 1); lay.addWidget(self.btn_refresh)

        # Queue Actions
        q_row = QHBoxLayout()
        self.btn_sweep = QPushButton("Generate In-App Queue")
        self.btn_sweep.clicked.connect(lambda: self.queue_sweep_clicked(background=False))
        
        self.btn_bg_sweep = QPushButton("🚀 Submit Background Job (Nohup)")
        self.btn_bg_sweep.setStyleSheet("background-color: #E91E63; color: white; font-weight: bold;")
        self.btn_bg_sweep.clicked.connect(lambda: self.queue_sweep_clicked(background=True))
        
        self.queue_list = QListWidget(); self.queue_list.setMaximumHeight(80)
        
        q_row.addWidget(self.btn_sweep); q_row.addWidget(self.btn_bg_sweep)
        lay.addLayout(q_row); lay.addWidget(self.queue_list)

        # Scoring
        row2 = QHBoxLayout()
        self.score_col = QLineEdit(f"score_{time.strftime('%H%M')}")
        self.btn_score = QPushButton("Score Selected"); self.btn_score.clicked.connect(self.score_clicked)
        row2.addWidget(QLabel("Score col:")); row2.addWidget(self.score_col); row2.addWidget(self.btn_score)
        lay.addLayout(row2)

        self.log = QTextEdit(); self.log.setReadOnly(True); lay.addWidget(self.log, 2)

    def _log(self, s): self.log.append(str(s))
    def _on_job_started(self, n): 
        if i := self.queue_list.findItems(n, Qt.MatchContains): i[0].setText(f"🔄 {n}"); i[0].setBackground(Qt.yellow)
    def _on_job_finished(self, n, s): 
        if i := self.queue_list.findItems(n, Qt.MatchContains): i[0].setText(f"{'✅' if s else '❌'} {n}"); i[0].setBackground(Qt.green if s else Qt.red)

    def browse_resume(self):
        p, _ = QFileDialog.getOpenFileName(self, "Checkpoint", self.annotate_tab.root_dir or "", "*.pt")
        if p: self.resume_path.setText(p)

    def _refresh_file_table(self):
        if not self.annotate_tab.root_dir: return
        from app.data.discover import discover_hdf5s
        files = [fp for _, fp in discover_hdf5s(self.annotate_tab.root_dir)]
        self.file_table.setRowCount(len(files))
        
        # Load saved states if available
        saved_files = self.settings.value("file_states", {})
        
        for r, fp in enumerate(files):
            fname = os.path.basename(fp)
            # Default state
            checked = 0
            mode = "use_label_col"
            
            # Apply saved state if matches
            if fname in saved_files:
                checked_bool, saved_mode = saved_files[fname]
                checked = 2 if checked_bool else 0
                mode = saved_mode

            it = QTableWidgetItem(""); it.setCheckState(checked)
            self.file_table.setItem(r, 0, it)
            self.file_table.setItem(r, 1, QTableWidgetItem(fp))
            
            cb = QComboBox(); cb.addItems(["use_label_col", "fixed_0", "fixed_1", "unlabeled (all rows)"])
            cb.setCurrentText(mode)
            self.file_table.setCellWidget(r, 2, cb)

    def _save_ui_state(self):
        # Save file table states
        file_states = {}
        for r in range(self.file_table.rowCount()):
            fname = os.path.basename(self.file_table.item(r, 1).text())
            checked = (self.file_table.item(r, 0).checkState() == 2)
            mode = self.file_table.cellWidget(r, 2).currentText()
            file_states[fname] = (checked, mode)
        self.settings.setValue("file_states", file_states)
        
        # Save Spinners/Inputs
        self.settings.setValue("cluster_col", self.cluster_col.text())
        self.settings.setValue("epochs", self.epochs.value())
        self.settings.setValue("pl_iters", self.pl_iters.value())
        self.settings.setValue("pl_thresh", self.pl_thresh.value())
        
        # NOTE: MultiCheckDropdown saving would require exposing `setChecked` on the widget
        # For now, we persist standard inputs which is the biggest pain point.

    def _load_ui_state(self):
        # Trigger file refresh which will read the setting we just wrote in _refresh_file_table
        self._refresh_file_table()
        
        # Load Inputs
        if v := self.settings.value("cluster_col"): self.cluster_col.setText(str(v))
        if v := self.settings.value("epochs"): self.epochs.setValue(float(v))
        if v := self.settings.value("pl_iters"): self.pl_iters.setValue(float(v))
        if v := self.settings.value("pl_thresh"): self.pl_thresh.setValue(float(v))

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

    def queue_sweep_clicked(self, background=False):
        # Save state before running
        self._save_ui_state()
        
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

        runs_root = getattr(self.cfg, "runs_dir", None) or os.path.join(self.annotate_tab.root_dir, "runs")
        combs = list(itertools.product(models, inputs, methods))
        
        job_definitions = []

        for arch, inp, meth in combs:
            kind, tname = ("timm_frozen", arch.split(":")[1]) if "timm" in arch else ("simple_cnn", "resnet18")
            job_name = f"{arch} | {inp} | {meth}"
            
            cfg = _make_train_config(
                model_kind=kind, timm_name=tname, inputs_mode=inp, training_method=meth,
                epochs=int(self.epochs.value()), batch_size=256, device="cuda", 
                image_key=self.cfg.image_key, aug_flags=tuple(self.aug_menu.selected()), 
                max_blur_sigma=self.blur_sigma.value()
            )
            ch = base_ch + (1 if inp == "image_and_mask" else 0)
            
            # Prepare data splits
            rng = np.random.default_rng(0); idx = rng.permutation(len(labeled)); n_val = int(len(labeled)*self.test_frac.value())
            val_items = [labeled[i] for i in idx[:n_val]]; train_items = [labeled[i] for i in idx[n_val:]]

            job_def = {
                "job_name": job_name,
                "cfg": cfg.__dict__,
                "method": meth,
                "in_chans": ch,
                "timm_name": tname,
                "resume_checkpoint": self.resume_path.text().strip() or None,
                "pl_iters": int(self.pl_iters.value()),
                "pl_thresh": self.pl_thresh.value(),
                # We pickle the items list directly
                "train_items": train_items,
                "val_items": val_items,
                "labeled_items": labeled,
                "unlabeled_items": unlabeled
            }
            job_definitions.append(job_def)

        if background:
            # Serialise jobs to a temp file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs(runs_root, exist_ok=True)
            config_pkl = os.path.join(runs_root, f"bg_sweep_{timestamp}.pkl")
            log_file = os.path.join(runs_root, f"bg_sweep_{timestamp}.log")
            
            payload = {"jobs": job_definitions, "runs_root": runs_root}
            with open(config_pkl, "wb") as f:
                pickle.dump(payload, f)
                
            cmd = f"nohup python -m app.training.headless_main --config '{config_pkl}' > '{log_file}' 2>&1 &"
            self._log(f"Launching background process: {cmd}")
            subprocess.Popen(cmd, shell=True)
            QMessageBox.information(self, "Launched", f"Jobs running in background.\nLog: {log_file}\n\nYou can close the GUI now.")
        
        else:
            # In-App Queue
            self._log(f"Generated {len(combs)} in-app jobs.")
            for j in job_definitions:
                self.queue_list.addItem(f"⏳ {j['job_name']}")
                
                # Create closure to capture current job definition 'j'
                def make_job(job_data):
                    t_cfg = TrainConfig(**job_data["cfg"])
                    if job_data["method"] == "pseudo-labeling":
                        return lambda cb: run_pseudo_label_job(
                            t_cfg, job_data["job_name"], job_data["labeled_items"], job_data["unlabeled_items"], 
                            job_data["timm_name"], job_data["in_chans"], runs_root, 
                            job_data["pl_iters"], job_data["pl_thresh"], cb
                        )
                    else:
                        return lambda cb: run_standard_job(
                            t_cfg, job_data["job_name"], job_data["train_items"], job_data["val_items"], 
                            job_data["timm_name"], job_data["in_chans"], runs_root, cb, 
                            resume_checkpoint=job_data["resume_checkpoint"]
                        )

                self.queue_manager.add_job(j['job_name'], make_job(j))

    def score_clicked(self):
        # (Same as before)
        if not self.model: return QMessageBox.warning(self, "No model", "Load/Train model first.")
        paths = self.annotate_tab.selected_paths
        if not paths: return QMessageBox.warning(self, "No files", "Select files.")
        run_in_thread(lambda cb: run_scoring_job(self.model, paths, self.score_col.text(), self.cfg, cb), 
                      parent=self, on_log=self._log, on_done=lambda: self._log("Scoring done."))
    
    def load_checkpoint_clicked(self):
        from PyQt5.QtWidgets import QFileDialog
        d = self.annotate_tab.root_dir or ""
        p, _ = QFileDialog.getOpenFileName(self, "Load Checkpoint", d, "*.pt *.pth")
        if p:
            try:
                self.model, kind, ch = load_model_from_checkpoint(p)
                self._log(f"Loaded {kind} from {p}"); QMessageBox.information(self, "Loaded", "Model Ready")
            except Exception as e:
                self._log(f"Error: {e}")