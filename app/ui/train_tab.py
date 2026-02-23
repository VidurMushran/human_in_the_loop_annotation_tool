# app/ui/train_tab.py
from __future__ import annotations

import os
import time
import itertools
import numpy as np
import pandas as pd
import h5py  # Added for channel detection
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QMessageBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QListWidget
)
from PyQt5.QtCore import Qt

from ..ui.widgets.multicheck_dropdown import MultiCheckDropdown
from ..utils.qt_threading import run_in_thread, JobQueueManager
from ..ml.models import SimpleCNN, make_timm_frozen_linear
from ..ml.train import TrainConfig, train_model
from ..ml.score import score_h5_file
from ..ml.dataset import Item

from ..experiments.registry import new_run_dir, write_run_yaml, write_metrics, write_checkpoint


def _basename(p: str) -> str:
    return os.path.basename(p) if p else p

def _make_train_config(**kwargs) -> TrainConfig:
    return TrainConfig(**kwargs)

class TrainTab(QWidget):
    def __init__(self, cfg, annotate_tab):
        super().__init__()
        self.cfg = cfg
        self.annotate_tab = annotate_tab
        self.model = None
        self.last_run_dir: Path | None = None
        
        # Initialize the Queue Manager
        self.queue_manager = JobQueueManager(self)
        self.queue_manager.log_signal.connect(self._log)
        self.queue_manager.job_started.connect(self._on_job_started)
        self.queue_manager.job_finished.connect(self._on_job_finished)
        self.queue_manager.queue_empty.connect(lambda: self._log("Queue is now empty. All sweeps finished!"))
        
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        # --- Sweep Configuration Row ---
        row = QHBoxLayout()

        self.model_menu = MultiCheckDropdown("Architectures")
        self.model_menu.add_option("simple_cnn", True)
        self.model_menu.add_option("timm:resnet18", False)
        self.model_menu.add_option("timm:resnet50", False)

        self.inputs_menu = MultiCheckDropdown("Input Modalities")
        self.inputs_menu.add_option("image_only", True)
        self.inputs_menu.add_option("image_and_mask", False)
        self.inputs_menu.add_option("image_and_features", False)

        self.method_menu = MultiCheckDropdown("Training Method")
        self.method_menu.add_option("supervised", True)
        self.method_menu.add_option("self-supervised", False)
        self.method_menu.add_option("pseudo-labeling", False) 

        self.aug_menu = MultiCheckDropdown("Augmentations")
        self.aug_menu.add_option("hflip", True)
        self.aug_menu.add_option("vflip", False)
        self.aug_menu.add_option("rotate90", True)
        self.aug_menu.add_option("gaussian_blur", False)

        row.addWidget(self.model_menu)
        row.addWidget(self.inputs_menu)
        row.addWidget(self.method_menu)
        row.addWidget(self.aug_menu)

        lay.addLayout(row)

        # --- Settings Row ---
        rowS = QHBoxLayout()
        self.cluster_col = QLineEdit("cluster")
        
        self.blur_sigma = QDoubleSpinBox()
        self.blur_sigma.setRange(0.0, 10.0)
        self.blur_sigma.setValue(1.5)

        self.test_frac = QDoubleSpinBox()
        self.test_frac.setRange(0.05, 0.5)
        self.test_frac.setValue(0.20)

        self.val_frac = QDoubleSpinBox()
        self.val_frac.setRange(0.05, 0.5)
        self.val_frac.setValue(0.20)
        
        # PL Settings
        self.pl_iters = QDoubleSpinBox()
        self.pl_iters.setRange(1, 10)
        self.pl_iters.setValue(3)
        self.pl_iters.setDecimals(0)
        
        self.pl_thresh = QDoubleSpinBox()
        self.pl_thresh.setRange(0.5, 0.99)
        self.pl_thresh.setSingleStep(0.05)
        self.pl_thresh.setValue(0.90)

        rowS.addWidget(QLabel("Max blur σ:"))
        rowS.addWidget(self.blur_sigma)
        rowS.addWidget(QLabel("Cluster col:"))
        rowS.addWidget(self.cluster_col)
        rowS.addWidget(QLabel("Test frac:"))
        rowS.addWidget(self.test_frac)
        rowS.addWidget(QLabel("Val frac:"))
        rowS.addWidget(self.val_frac)
        
        rowS.addWidget(QLabel("PL Iters:"))
        rowS.addWidget(self.pl_iters)
        rowS.addWidget(QLabel("PL Thresh:"))
        rowS.addWidget(self.pl_thresh)
        
        lay.addLayout(rowS)

        # --- File Selection Table ---
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(3)
        self.file_table.setHorizontalHeaderLabels(["Use in Training?", "HDF5 file", "Label mode"])
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        lay.addWidget(QLabel("Training file selection:"))
        lay.addWidget(self.file_table, 1)

        self.btn_refresh_files = QPushButton("Refresh file list from root")
        self.btn_refresh_files.clicked.connect(self._refresh_file_table)
        lay.addWidget(self.btn_refresh_files)

        # --- Queue UI and Generate Sweep ---
        queue_row = QHBoxLayout()
        self.btn_queue_sweep = QPushButton("Generate Sweep & Add to Queue")
        self.btn_queue_sweep.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.btn_queue_sweep.clicked.connect(self.queue_sweep_clicked)
        
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(100)
        
        queue_layout = QVBoxLayout()
        queue_layout.addWidget(self.btn_queue_sweep)
        queue_layout.addWidget(QLabel("Job Queue Monitor:"))
        queue_layout.addWidget(self.queue_list)
        
        lay.addLayout(queue_row)
        lay.addLayout(queue_layout)

        # --- Scoring row ---
        row2 = QHBoxLayout()
        self.score_col = QLineEdit(f"model_score_{time.strftime('%Y%m%d_%H%M%S')}")
        self.btn_load_ckpt = QPushButton("Load Model Checkpoint")
        self.btn_load_ckpt.clicked.connect(self.load_checkpoint_clicked)
        self.btn_score = QPushButton("Score selected HDF5s -> write score column")
        self.btn_score.clicked.connect(self.score_clicked)
        row2.addWidget(QLabel("Score column:"))
        row2.addWidget(self.score_col)
        row2.addWidget(self.btn_load_ckpt)
        row2.addWidget(self.btn_score)
        lay.addLayout(row2)

        # --- Log ---
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, 2)

    def _log(self, s: str):
        self.log.append(str(s))
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_job_started(self, name: str):
        items = self.queue_list.findItems(name, Qt.MatchContains)
        if items:
            items[0].setText(f"🔄 [RUNNING] {name}")
            items[0].setBackground(Qt.yellow)

    def _on_job_finished(self, name: str, success: bool):
        items = self.queue_list.findItems(name, Qt.MatchContains)
        if items:
            if success:
                items[0].setText(f"✅ [DONE] {name}")
                items[0].setBackground(Qt.green)
            else:
                items[0].setText(f"❌ [FAILED] {name}")
                items[0].setBackground(Qt.red)

    def _create_run_dir(self, suffix: str = "") -> Path:
        root = self.annotate_tab.root_dir
        if not root:
            raise RuntimeError("No root_dir set")

        runs_root = getattr(self.cfg, "runs_dir", None) or os.path.join(root, "runs")
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{suffix}" if suffix else timestamp
        
        run_dir = runs_root / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _refresh_file_table(self):
        if not self.annotate_tab.root_dir:
            return

        from app.data.discover import discover_hdf5s

        files = [fp for _, fp in discover_hdf5s(self.annotate_tab.root_dir)]
        self.file_table.setRowCount(len(files))

        for r, fp in enumerate(files):
            it0 = QTableWidgetItem("")
            it0.setCheckState(0) 
            self.file_table.setItem(r, 0, it0)
            self.file_table.setItem(r, 1, QTableWidgetItem(fp))

            cb = QComboBox()
            cb.addItems(["use_label_col", "fixed_0", "fixed_1", "unlabeled (all rows)"])
            self.file_table.setCellWidget(r, 2, cb)

        self._log(f"Loaded {len(files)} file(s) into training selection table.")

    def _get_supervised_file_specs(self):
        specs = []
        for r in range(self.file_table.rowCount()):
            it = self.file_table.item(r, 0)
            if it is None:
                continue
            use = it.checkState() == 2
            if not use:
                continue
            fp = self.file_table.item(r, 1).text()
            mode = self.file_table.cellWidget(r, 2).currentText()
            specs.append((fp, mode))
        return specs

    def _items_from_file(self, fp: str, mode: str, label_col: str, cluster_col: str):
        try:
            df = pd.read_hdf(fp, key=self.cfg.features_key)
        except Exception:
            return []

        if mode == "fixed_0":
            idxs = np.arange(len(df), dtype=int)
            y = np.zeros(len(df), dtype=int)
        elif mode == "fixed_1":
            idxs = np.arange(len(df), dtype=int)
            y = np.ones(len(df), dtype=int)
        else:
            if label_col not in df.columns:
                return []
            s = df[label_col].astype(str).str.lower()
            junk_idx = np.where(s.str.contains("junk"))[0]
            cell_idx = np.where(s.str.contains("cell"))[0]
            idxs = np.concatenate([junk_idx, cell_idx]).astype(int)
            y = np.concatenate([
                np.ones(len(junk_idx), dtype=int),
                np.zeros(len(cell_idx), dtype=int),
            ])

        cl = np.full(len(idxs), -1, dtype=int)
        if cluster_col and cluster_col in df.columns:
            raw = df.loc[df.index[idxs], cluster_col].to_numpy()
            raw = np.where(pd.isna(raw), -1, raw)
            try:
                cl = raw.astype(int)
            except Exception:
                cl = np.array([hash(x) % (10**9) for x in raw], dtype=int)

        return [Item(fp, int(i), int(yy), int(cc)) for i, yy, cc in zip(idxs, y, cl)]

    def _strat_keys(self, items):
        keys = []
        for it in items:
            if getattr(it, "cluster", -1) is not None and int(it.cluster) >= 0:
                keys.append((int(it.y), int(it.cluster)))
            else:
                keys.append((int(it.y), -1))
        return np.array([hash(k) for k in keys], dtype=np.int64)

    def load_checkpoint_clicked(self):
        from PyQt5.QtWidgets import QFileDialog
        import torch
        from ..ml.models import SimpleCNN, make_timm_frozen_linear

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
                # Assuming 4 channels as default if not in config
                model = SimpleCNN(in_ch=4, n_classes=2)
            else:
                model = make_timm_frozen_linear(timm_name, in_chans=4, n_classes=2)

            model.load_state_dict(ckpt["model_state_dict"])
            self.model = model
            
            self._log(f"Successfully loaded {kind} model from {path}")
            QMessageBox.information(self, "Success", f"Model loaded from:\n{path}")
            
        except Exception as e:
            self._log(f"[error] Failed to load checkpoint: {e}")
            QMessageBox.warning(self, "Load Error", f"Failed to load checkpoint:\n{e}")

    def queue_sweep_clicked(self):
        if not self.annotate_tab.root_dir:
            QMessageBox.information(self, "No root", "Pick a root directory in the Annotate tab first.")
            return

        specs = self._get_supervised_file_specs()
        if not specs:
            QMessageBox.warning(self, "No files selected", "Select at least one file in the table.")
            return

        # --- AUTO-DETECT CHANNELS ---
        # Inspect first file to determine base channels (usually 3 or 4)
        base_channels = 3 # fallback default
        if specs:
            try:
                first_fp = specs[0][0]
                with h5py.File(first_fp, 'r') as f:
                    # dataset.py assumes HWC, so we check last dimension
                    if self.cfg.image_key in f:
                        shape = f[self.cfg.image_key].shape
                        if len(shape) == 4:
                            base_channels = shape[-1]
                        self._log(f"Auto-detected base channels from {os.path.basename(first_fp)}: {base_channels}")
            except Exception as e:
                self._log(f"[warn] Could not auto-detect channels: {e}. Defaulting to 3.")
        # ----------------------------

        selected_models = self.model_menu.selected()
        selected_inputs = self.inputs_menu.selected()
        selected_methods = self.method_menu.selected()
        base_augs = tuple(self.aug_menu.selected())
        
        if not selected_models or not selected_inputs or not selected_methods:
            QMessageBox.warning(self, "Missing Selections", "Must select at least one architecture, input, and method!")
            return

        label_col = self.annotate_tab.label_col
        cluster_col = self.cluster_col.text().strip()
        
        initial_labeled_items = []
        unlabeled_items_pool = []
        per_file_counts = {}

        for fp, mode in specs:
            if mode == "unlabeled (all rows)":
                try:
                    df = pd.read_hdf(fp, key=self.cfg.features_key)
                    idxs = np.arange(len(df), dtype=int)
                    cl = np.full(len(idxs), -1, dtype=int)
                    if cluster_col and cluster_col in df.columns:
                        raw = df[cluster_col].to_numpy()
                        raw = np.where(pd.isna(raw), -1, raw)
                        cl = np.array([hash(x) % (10**9) for x in raw], dtype=int)
                    
                    unlabeled_items_pool.extend([Item(fp, i, -1, c) for i, c in zip(idxs, cl)])
                except Exception as e:
                    self._log(f"[warn] Failed to read unlabeled rows from {fp}: {e}")
            else:
                file_items = self._items_from_file(fp, mode, label_col=label_col, cluster_col=cluster_col)
                if file_items:
                    initial_labeled_items.extend(file_items)
                    y_arr = np.array([it.y for it in file_items], dtype=int)
                    per_file_counts[fp] = {
                        "mode": mode, "n": len(file_items), "n0": int((y_arr == 0).sum()), "n1": int((y_arr == 1).sum()),
                    }

        if len(initial_labeled_items) < 10 and "supervised" in selected_methods:
            QMessageBox.warning(self, "Too few labels", "Found very few labeled rows. Supervised training may fail.")

        rng = np.random.default_rng(0)
        idx = rng.permutation(len(initial_labeled_items))
        n_test = int(float(self.test_frac.value()) * len(initial_labeled_items))
        test_items = [initial_labeled_items[i] for i in idx[:n_test]]
        remain = [initial_labeled_items[i] for i in idx[n_test:]]
        n_val = int(float(self.val_frac.value()) * len(remain))
        val_items = remain[:n_val]
        train_items = remain[n_val:]

        combinations = list(itertools.product(selected_models, selected_inputs, selected_methods))
        self._log(f"Generated {len(combinations)} sweep combinations.")

        for (arch, inputs_mode, method) in combinations:
            
            if arch.startswith("timm:"):
                kind = "timm_frozen"
                timm_name = arch.split(":")[1]
            else:
                kind = "simple_cnn"
                timm_name = "resnet18" 

            job_name = f"{arch} | {inputs_mode} | {method}"
            self.queue_list.addItem(f"⏳ [PENDING] {job_name}")

            cfg = _make_train_config(
                model_kind=kind,
                timm_name=timm_name,
                inputs_mode=inputs_mode,         
                training_method=method,          
                epochs=5,
                batch_size=256,
                lr=1e-3,
                device="cuda",
                image_key=self.cfg.image_key,
                mask_key="masks",                
                target_hw=75,
                aug_flags=base_augs,
                max_blur_sigma=float(self.blur_sigma.value()),
                seed=0,
            )

            # Determine dynamic input channels
            # If base is 4, image_only=4. If image_and_mask, 4+1=5.
            dynamic_in_channels = base_channels
            if inputs_mode == "image_and_mask":
                dynamic_in_channels += 1

            if method == "pseudo-labeling":
                pl_iterations = int(self.pl_iters.value())
                pl_threshold = float(self.pl_thresh.value())

                def build_pl_closure(current_cfg, j_name, init_labeled, init_unlabeled, t_name, in_ch):
                    def _job(log_cb):
                        log_cb(f"Starting Pseudo-Labeling job: {j_name}")
                        
                        import copy
                        current_train_pool = copy.deepcopy(init_labeled)
                        current_unlabeled_pool = copy.deepcopy(init_unlabeled)
                        
                        rng = np.random.default_rng(0)
                        idx = rng.permutation(len(current_train_pool))
                        n_val = int(0.20 * len(current_train_pool))
                        current_val_pool = [current_train_pool[i] for i in idx[:n_val]]
                        current_train_pool = [current_train_pool[i] for i in idx[n_val:]]
                        
                        for iter_idx in range(pl_iterations):
                            log_cb(f"\n--- PL Iteration {iter_idx+1}/{pl_iterations} ---")
                            log_cb(f"Train size: {len(current_train_pool)} | Unlabeled pool: {len(current_unlabeled_pool)}")
                            
                            # Use DYNAMIC channels
                            if current_cfg.model_kind == "simple_cnn":
                                model = SimpleCNN(in_ch=in_ch, n_classes=2)
                            else:
                                model = make_timm_frozen_linear(t_name, in_chans=in_ch, n_classes=2)
                            
                            current_cfg.training_method = "supervised"
                            trained_model, best = train_model(model, current_train_pool, current_val_pool, current_cfg, log_cb=log_cb)
                            self.model = trained_model
                            
                            if iter_idx == pl_iterations - 1 or len(current_unlabeled_pool) == 0:
                                safe_arch = t_name if current_cfg.model_kind == "timm_frozen" else "simple_cnn"
                                suffix = f"{safe_arch}_{current_cfg.inputs_mode}_PL"
                                run_dir = self._create_run_dir(suffix=suffix)
                                self.last_run_dir = run_dir
                                extra = {"job_name": j_name, "mode": "pseudo-labeling", "n_train_final": len(current_train_pool)}
                                write_run_yaml(run_dir, cfg=current_cfg.__dict__, extra=extra)
                                
                                import torch
                                ckpt_src = run_dir / "checkpoint_src.pt"
                                torch.save({"model_state_dict": trained_model.state_dict(), "train_config": current_cfg.__dict__, "best": best}, ckpt_src)
                                write_checkpoint(run_dir, str(ckpt_src))
                                metrics = dict(best) if isinstance(best, dict) else {"best": str(best)}
                                write_metrics(run_dir, metrics)
                                log_cb(f"Finished. Final run saved to {run_dir}")
                                break
                                
                            log_cb("Scoring unlabeled pool...")
                            trained_model.eval()
                            trained_model.to(current_cfg.device)
                            
                            from app.ml.dataset import H5StreamDataset
                            from torch.utils.data import DataLoader
                            import torch
                            
                            u_ds = H5StreamDataset(current_unlabeled_pool, image_key=current_cfg.image_key, 
                                                   target_hw=current_cfg.target_hw, aug_flags=(), 
                                                   inputs_mode=current_cfg.inputs_mode, training_method="supervised")
                            u_dl = DataLoader(u_ds, batch_size=256, shuffle=False, num_workers=2)
                            
                            newly_labeled = []
                            remaining_unlabeled = []
                            
                            global_idx = 0
                            with torch.no_grad():
                                for batch in u_dl:
                                    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                                    xb = xb.to(current_cfg.device)
                                    logits = trained_model(xb)
                                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                                    
                                    for p in probs:
                                        it = current_unlabeled_pool[global_idx]
                                        if p >= pl_threshold:
                                            newly_labeled.append(Item(it.h5_path, it.row_idx, 1, it.cluster))
                                        elif p <= (1.0 - pl_threshold):
                                            newly_labeled.append(Item(it.h5_path, it.row_idx, 0, it.cluster))
                                        else:
                                            remaining_unlabeled.append(it)
                                        global_idx += 1
                                        
                            log_cb(f"Found {len(newly_labeled)} high-confidence pseudo-labels.")
                            current_train_pool.extend(newly_labeled)
                            current_unlabeled_pool = remaining_unlabeled

                    return _job
                
                run_func = build_pl_closure(cfg, job_name, initial_labeled_items, unlabeled_items_pool, timm_name, dynamic_in_channels)
                self.queue_manager.add_job(job_name, run_func)

            else:
                def build_std_closure(current_cfg, j_name, t_items, v_items, t_name, in_ch):
                    def _job(log_cb):
                        log_cb(f"Starting sweep job: {j_name}")
                        
                        # Use DYNAMIC channels
                        if current_cfg.model_kind == "simple_cnn":
                            model = SimpleCNN(in_ch=in_ch, n_classes=2)
                        else:
                            model = make_timm_frozen_linear(t_name, in_chans=in_ch, n_classes=2)

                        safe_arch = t_name if current_cfg.model_kind == "timm_frozen" else "simple_cnn"
                        suffix = f"{safe_arch}_{current_cfg.inputs_mode}_{current_cfg.training_method}"
                        run_dir = self._create_run_dir(suffix=suffix)
                        self.last_run_dir = run_dir
                        extra = {"job_name": j_name, "mode": current_cfg.training_method, "inputs": current_cfg.inputs_mode, "n_train": len(t_items)}
                        write_run_yaml(run_dir, cfg=current_cfg.__dict__, extra=extra)

                        trained_model, best = train_model(model, t_items, v_items, current_cfg, log_cb=log_cb)
                        self.model = trained_model 

                        try:
                            import torch
                            ckpt_src = run_dir / "checkpoint_src.pt"
                            torch.save({"model_state_dict": trained_model.state_dict(), "train_config": current_cfg.__dict__, "best": best, "extra": extra}, ckpt_src)
                            write_checkpoint(run_dir, str(ckpt_src))
                            metrics = dict(best) if isinstance(best, dict) else {"best": str(best)}
                            metrics.pop("state", None)
                            write_metrics(run_dir, metrics)
                        except Exception as e:
                            log_cb(f"Warning: Artifact save failed: {e}")

                    return _job
                
                run_func = build_std_closure(cfg, job_name, train_items, val_items, timm_name, dynamic_in_channels)
                self.queue_manager.add_job(job_name, run_func)

    def score_clicked(self):
        if self.model is None:
            QMessageBox.information(self, "No model", "Train a model first.")
            return

        paths = self.annotate_tab.selected_paths
        if not paths:
            QMessageBox.information(self, "No files", "Select at least one HDF5 in the Annotate tab.")
            return

        score_col = self.score_col.text().strip()
        if not score_col:
            return

        self._log(f"Scoring {len(paths)} file(s) into column: {score_col}")

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
                    log_cb(f"[ERROR] Failed to score {_basename(fp)}: {e}")

        run_in_thread(_score_job, parent=self, on_log=self._log, on_error=lambda e: self._log(e), on_done=lambda: self._log("Scoring thread finished."))