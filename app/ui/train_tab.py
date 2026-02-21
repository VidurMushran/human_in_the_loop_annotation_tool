from __future__ import annotations

import os
import time
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QMessageBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView
)

from ..ui.widgets.multicheck_dropdown import MultiCheckDropdown
from ..utils.qt_threading import run_in_thread
from ..ml.models import SimpleCNN, make_timm_frozen_linear
from ..ml.train import TrainConfig, train_supervised
from ..ml.score import score_h5_file
from ..ml.dataset import Item

from ..experiments.registry import new_run_dir, write_run_yaml, write_metrics, write_checkpoint


def _basename(p: str) -> str:
    return os.path.basename(p) if p else p


def _make_train_config(**kwargs) -> TrainConfig:
    # Keep strict: TrainConfig is ours
    return TrainConfig(**kwargs)


class TrainTab(QWidget):
    def __init__(self, cfg, annotate_tab):
        super().__init__()
        self.cfg = cfg
        self.annotate_tab = annotate_tab
        self.model = None
        self.last_run_dir: Path | None = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        # --- Model + Aug row ---
        row = QHBoxLayout()

        self.model_kind = QComboBox()
        self.model_kind.addItems(["simple_cnn", "timm_frozen"])
        self.timm_name = QLineEdit("resnet18")

        self.aug_menu = MultiCheckDropdown("Augmentations")
        self.aug_menu.add_option("hflip", True)
        self.aug_menu.add_option("vflip", False)
        self.aug_menu.add_option("rotate90", True)
        self.aug_menu.add_option("gaussian_blur", False)

        self.blur_sigma = QDoubleSpinBox()
        self.blur_sigma.setRange(0.0, 10.0)
        self.blur_sigma.setSingleStep(0.25)
        self.blur_sigma.setValue(1.5)

        self.btn_train = QPushButton("Train (supervised)")
        self.btn_train.clicked.connect(self.train_clicked)

        row.addWidget(QLabel("Model:"))
        row.addWidget(self.model_kind)
        row.addWidget(QLabel("timm name (if frozen):"))
        row.addWidget(self.timm_name)
        row.addWidget(self.aug_menu)
        row.addWidget(QLabel("Max blur σ:"))
        row.addWidget(self.blur_sigma)
        row.addWidget(self.btn_train)
        lay.addLayout(row)

        # --- Split stratification config ---
        rowS = QHBoxLayout()
        self.cluster_col = QLineEdit("cluster")  # optional column in features

        self.test_frac = QDoubleSpinBox()
        self.test_frac.setRange(0.05, 0.5)
        self.test_frac.setSingleStep(0.05)
        self.test_frac.setValue(0.20)

        self.val_frac = QDoubleSpinBox()
        self.val_frac.setRange(0.05, 0.5)
        self.val_frac.setSingleStep(0.05)
        self.val_frac.setValue(0.20)

        rowS.addWidget(QLabel("Cluster col (optional):"))
        rowS.addWidget(self.cluster_col)
        rowS.addWidget(QLabel("Test frac:"))
        rowS.addWidget(self.test_frac)
        rowS.addWidget(QLabel("Val frac (of remaining):"))
        rowS.addWidget(self.val_frac)
        lay.addLayout(rowS)

        # --- Supervised file selection table ---
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(3)
        self.file_table.setHorizontalHeaderLabels(["Supervised?", "HDF5 file", "Label mode"])
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        lay.addWidget(QLabel("Training file selection (supervised):"))
        lay.addWidget(self.file_table, 1)

        self.btn_refresh_files = QPushButton("Refresh file list from root")
        self.btn_refresh_files.clicked.connect(self._refresh_file_table)
        lay.addWidget(self.btn_refresh_files)

        # --- Scoring row ---
        row2 = QHBoxLayout()
        self.score_col = QLineEdit(f"model_score_{time.strftime('%Y%m%d_%H%M%S')}")
        
        from PyQt5.QtWidgets import QFileDialog
        self.btn_load_ckpt = QPushButton("Load Model Checkpoint")
        self.btn_load_ckpt.clicked.connect(self.load_checkpoint_clicked)

        self.btn_score = QPushButton("Score selected HDF5s -> write score column")
        self.btn_score.clicked.connect(self.score_clicked)
        
        row2.addWidget(QLabel("Score column:"))
        row2.addWidget(self.score_col)
        row2.addWidget(self.btn_load_ckpt) # Added to layout
        row2.addWidget(self.btn_score)
        lay.addLayout(row2)

        # --- Log ---
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, 2)

    def _log(self, s: str):
        self.log.append(str(s))

    def _create_run_dir(self) -> Path:
        """
        Must return Path because registry expects Path.
        Prefer cfg.runs_dir if present; else <root>/runs.
        """
        root = self.annotate_tab.root_dir
        if not root:
            raise RuntimeError("No root_dir set")

        runs_root = getattr(self.cfg, "runs_dir", None) or os.path.join(root, "runs")
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        run_dir = new_run_dir(str(runs_root))
        return run_dir

    def _refresh_file_table(self):
        if not self.annotate_tab.root_dir:
            return

        from app.data.discover import discover_hdf5s

        files = [fp for _, fp in discover_hdf5s(self.annotate_tab.root_dir)]
        self.file_table.setRowCount(len(files))

        for r, fp in enumerate(files):
            it0 = QTableWidgetItem("")
            it0.setCheckState(0)  # unchecked
            self.file_table.setItem(r, 0, it0)

            self.file_table.setItem(r, 1, QTableWidgetItem(fp))

            cb = QComboBox()
            cb.addItems(["use_label_col", "fixed_0", "fixed_1"])
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
        import pandas as pd

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
        # stratify by (y, cluster) if cluster >=0 else by y only
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

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", str(self.annotate_tab.root_dir or ""), "PyTorch Checkpoints (*.pt *.pth)"
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
            
            self._log(f"Successfully loaded {kind} model from {path}")
            QMessageBox.information(self, "Success", f"Model loaded from:\n{path}")
            
        except Exception as e:
            self._log(f"[error] Failed to load checkpoint: {e}")
            QMessageBox.warning(self, "Load Error", f"Failed to load checkpoint:\n{e}")

    def train_clicked(self):
        if not self.annotate_tab.root_dir:
            QMessageBox.information(self, "No root", "Pick a root directory in the Annotate tab first.")
            return

        specs = self._get_supervised_file_specs()
        if not specs:
            QMessageBox.warning(self, "No supervised files selected",
                                "Select at least one file in the table (Supervised? column).")
            return

        label_col = self.annotate_tab.label_col
        cluster_col = self.cluster_col.text().strip()

        items: list[Item] = []
        per_file_counts = {}

        for fp, mode in specs:
            file_items = self._items_from_file(fp, mode, label_col=label_col, cluster_col=cluster_col)
            if not file_items:
                self._log(f"[warn] No usable labeled rows from: {_basename(fp)} (mode={mode})")
                continue
            items.extend(file_items)
            y_arr = np.array([it.y for it in file_items], dtype=int)
            per_file_counts[fp] = {
                "mode": mode,
                "n": len(file_items),
                "n0": int((y_arr == 0).sum()),
                "n1": int((y_arr == 1).sum()),
            }

        if len(items) < 200:
            QMessageBox.warning(self, "Too few labels", f"Only found {len(items)} labeled rows.")
            return

        self._log("Supervised training files:")
        for fp, info in per_file_counts.items():
            self._log(f"  - {_basename(fp)} | mode={info['mode']} | n={info['n']} (y0={info['n0']}, y1={info['n1']})")

        # Stratified split
        try:
            from sklearn.model_selection import StratifiedShuffleSplit

            keys = self._strat_keys(items)
            test_frac = float(self.test_frac.value())
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=0)
            tr_idx, te_idx = next(sss1.split(np.zeros(len(items)), keys))

            remain = [items[i] for i in tr_idx]
            test_items = [items[i] for i in te_idx]

            val_frac = float(self.val_frac.value())
            keys2 = self._strat_keys(remain)
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=1)
            tr2_idx, va_idx = next(sss2.split(np.zeros(len(remain)), keys2))

            train_items = [remain[i] for i in tr2_idx]
            val_items = [remain[i] for i in va_idx]

        except Exception as e:
            self._log(f"[warn] Stratified split failed ({e}); using random split.")
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(items))
            n_test = int(float(self.test_frac.value()) * len(items))
            test_items = [items[i] for i in idx[:n_test]]
            remain = [items[i] for i in idx[n_test:]]
            n_val = int(float(self.val_frac.value()) * len(remain))
            val_items = remain[:n_val]
            train_items = remain[n_val:]

        kind = self.model_kind.currentText()
        timm_name = self.timm_name.text().strip()
        aug_flags = tuple(self.aug_menu.selected())
        max_blur_sigma = float(self.blur_sigma.value())

        if kind == "simple_cnn":
            model = SimpleCNN(in_ch=4, n_classes=2)
        else:
            model = make_timm_frozen_linear(timm_name, in_chans=4, n_classes=2)

        cfg = _make_train_config(
            model_kind=kind,
            timm_name=timm_name,
            epochs=5,
            batch_size=256,
            lr=1e-3,
            device="cuda",
            image_key=self.cfg.image_key,
            target_hw=75,
            aug_flags=aug_flags,
            max_blur_sigma=max_blur_sigma,
            seed=0,
        )

        # Registry-compatible run dir + yaml
        run_dir = self._create_run_dir()
        self.last_run_dir = run_dir

        extra = {
            "mode": "supervised",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label_col": label_col,
            "cluster_col": cluster_col,
            "features_key": self.cfg.features_key,
            "image_key": self.cfg.image_key,
            "train_files": [{"path": fp, **info} for fp, info in per_file_counts.items()],
            "n_items_total": int(len(items)),
            "n_train": int(len(train_items)),
            "n_val": int(len(val_items)),
            "n_test": int(len(test_items)),
            "aug_flags": list(aug_flags),
            "max_blur_sigma": max_blur_sigma,
        }

        try:
            write_run_yaml(run_dir, cfg=cfg.__dict__, extra=extra)
        except Exception as e:
            self._log(f"[warn] write_run_yaml failed: {e}")

        self._log(f"Run dir: {run_dir}")
        self._log(f"Split: train={len(train_items)} val={len(val_items)} test={len(test_items)}")
        self._log(f"Aug: {aug_flags if aug_flags else 'none'} | max_blur_sigma={max_blur_sigma}")

        def _train_job(log_cb):
            trained_model, best = train_supervised(model, train_items, val_items, cfg, log_cb=log_cb)
            self.model = trained_model

            metrics = dict(best) if isinstance(best, dict) else {"best": str(best)}
            metrics.pop("state", None)

            def _jsonify(x):
                import numpy as _np
                import torch as _torch
                if isinstance(x, _torch.Tensor):
                    return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
                if isinstance(x, (_np.integer, _np.floating)):
                    return x.item()
                if isinstance(x, dict):
                    return {k: _jsonify(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    return [_jsonify(v) for v in x]
                return x

            metrics = _jsonify(metrics)

            try:
                write_metrics(run_dir, metrics)
            except Exception as e:
                log_cb(f"[warn] write_metrics failed: {e}")

            # registry expects a *path* to copy as checkpoint.pt
            ckpt_src = run_dir / "checkpoint_src.pt"
            try:
                import torch
                torch.save(
                    {
                        "model_state_dict": trained_model.state_dict(),
                        "train_config": cfg.__dict__,
                        "best": best,
                        "extra": extra,
                    },
                    ckpt_src,
                )
                write_checkpoint(run_dir, str(ckpt_src))
            except Exception as e:
                log_cb(f"[warn] write_checkpoint failed: {e}")

            return run_dir

        def _on_done():
            self._log("Training thread finished.")
            if self.last_run_dir:
                self._log(f"Saved run artifacts in: {self.last_run_dir}")

        run_in_thread(
            _train_job,
            parent=self,
            on_log=self._log,
            on_error=lambda e: self._log(e),
            on_done=_on_done,
        )

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
                    # If one file fails, log the error but CONTINUE to the next file!
                    log_cb(f"[ERROR] Failed to score {_basename(fp)}: {e}")

        run_in_thread(
            _score_job,
            parent=self,
            on_log=self._log,
            on_error=lambda e: self._log(e),
            on_done=lambda: self._log("Scoring thread finished. You can now sort by the new score column."),
        )
