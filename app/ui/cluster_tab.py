# app/ui/cluster_tab.py
from __future__ import annotations
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, 
    QMessageBox, QFileDialog, QSplitter, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from app.utils.model_helpers import load_model_from_checkpoint
from app.data.h5io import read_images_by_indices, write_features_rows_inplace
from app.ui.widgets.gallery_pane import GalleryPane
from app.imaging.render import channels_to_rgb8bit

class ClusterTab(QWidget):
    def __init__(self, cfg, annotate_tab):
        super().__init__()
        self.cfg = cfg
        self.annotate_tab = annotate_tab
        self.embed_model = None
        self.score_model = None
        
        self.cluster_data = {} # {cluster_id: [ {fp, idx, score} ]}
        self.current_cluster = 0
        self.to_drop = set() # (fp, idx) to drop from labeling
        
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        
        # --- Controls ---
        row = QHBoxLayout()
        self.btn_load_emb = QPushButton("Load Embedding Model (SimCLR)"); self.btn_load_emb.clicked.connect(lambda: self.load_model('E'))
        self.lbl_emb = QLabel("Embed: None")
        
        self.btn_load_scr = QPushButton("Load Scoring Model"); self.btn_load_scr.clicked.connect(lambda: self.load_model('S'))
        self.lbl_scr = QLabel("Score: None")
        
        row.addWidget(self.btn_load_emb); row.addWidget(self.lbl_emb)
        row.addWidget(self.btn_load_scr); row.addWidget(self.lbl_scr)
        lay.addLayout(row)
        
        # --- Analysis Settings ---
        set_row = QHBoxLayout()
        self.spin_k = QSpinBox(); self.spin_k.setRange(2, 50); self.spin_k.setValue(10); self.spin_k.setPrefix("K=")
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.5, 1.0); self.spin_conf.setValue(0.99); self.spin_conf.setPrefix("Conf > ")
        self.btn_run = QPushButton("Run Clustering & Filter"); self.btn_run.clicked.connect(self.run_analysis)
        
        set_row.addWidget(self.spin_k); set_row.addWidget(self.spin_conf); set_row.addWidget(self.btn_run)
        lay.addLayout(set_row)
        
        # --- Main Gallery ---
        self.gallery = GalleryPane("Cluster View", self.on_tile_clicked)
        lay.addWidget(self.gallery, 1)
        
        # --- Actions ---
        act_row = QHBoxLayout()
        self.btn_prev = QPushButton("<< Prev Cluster"); self.btn_prev.clicked.connect(lambda: self.change_cluster(-1))
        self.lbl_cluster = QLabel("Cluster: -")
        self.btn_next = QPushButton("Next Cluster >>"); self.btn_next.clicked.connect(lambda: self.change_cluster(1))
        
        self.btn_label_junk = QPushButton("Label REMAINING as Junk"); self.btn_label_junk.setStyleSheet("background: #F44336; color: white;")
        self.btn_label_junk.clicked.connect(lambda: self.bulk_label(1))
        
        self.btn_label_cell = QPushButton("Label REMAINING as Cell"); self.btn_label_cell.setStyleSheet("background: #4CAF50; color: white;")
        self.btn_label_cell.clicked.connect(lambda: self.bulk_label(0))

        act_row.addWidget(self.btn_prev); act_row.addWidget(self.lbl_cluster); act_row.addWidget(self.btn_next)
        act_row.addStretch()
        act_row.addWidget(self.btn_label_junk); act_row.addWidget(self.btn_label_cell)
        lay.addLayout(act_row)

    def load_model(self, tag):
        p, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "*.pt")
        if not p: return
        try:
            m, _, _ = load_model_from_checkpoint(p, device="cuda")
            if tag == 'E': 
                self.embed_model = m; self.lbl_emb.setText(f"Embed: {os.path.basename(p)}")
            else: 
                self.score_model = m; self.lbl_scr.setText(f"Score: {os.path.basename(p)}")
        except Exception as e: QMessageBox.warning(self, "Error", str(e))

    def run_analysis(self):
        if not (self.embed_model and self.score_model): return QMessageBox.warning(self, "Error", "Load both models.")
        paths = self.annotate_tab.selected_paths
        if not paths: return QMessageBox.warning(self, "Error", "Select files.")
        
        k = self.spin_k.value()
        conf_thresh = self.spin_conf.value()
        
        # 1. Gather Data (Bulk IO)
        import h5py
        all_refs = []
        for fp in paths:
            try:
                with h5py.File(fp,'r') as f: n=f[self.cfg.image_key].shape[0]
                all_refs.extend([(fp, i) for i in range(n)])
            except: pass
            
        # Sample if too large (e.g. > 10k) to keep UI responsive
        if len(all_refs) > 10000:
            import random; all_refs = random.sample(all_refs, 10000)

        # 2. Inference
        embeddings = []
        scores = []
        valid_refs = []
        
        self.embed_model.eval(); self.score_model.eval()
        
        by_file = {}
        for fp, r in all_refs: by_file.setdefault(fp, []).append(r)
        
        with torch.no_grad():
            for fp, rows in by_file.items():
                # Process in chunks of 256
                rows_arr = np.array(rows)
                for i in range(0, len(rows), 256):
                    batch_idx = rows_arr[i:i+256]
                    try:
                        imgs = read_images_by_indices(fp, batch_idx, image_key=self.cfg.image_key)
                        xb = torch.from_numpy(imgs).permute(0,3,1,2).float().cuda()
                        
                        # Embed
                        emb = self.embed_model(xb).cpu().numpy()
                        
                        # Score
                        logits = self.score_model(xb)
                        if hasattr(self.score_model, "classifier"):
                             logits = self.score_model.classifier(logits)
                        probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                        
                        embeddings.append(emb)
                        scores.append(probs)
                        valid_refs.extend([(fp, r) for r in batch_idx])
                    except: pass
        
        if not embeddings: return
        
        X = np.concatenate(embeddings)
        S = np.concatenate(scores)
        
        # 3. Cluster
        print("Clustering...")
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # 4. Filter & Group
        self.cluster_data = {i: [] for i in range(k)}
        for idx, (fp, ridx) in enumerate(valid_refs):
            cid = labels[idx]
            score = S[idx]
            
            # Confidence Filter: Keep if Score > Thresh (Junk) OR Score < (1-Thresh) (Cell)
            # Or just highly confident predictions? The user asked for "0.99-1.0".
            # Assuming they want high confidence predictions of ANY class grouped by visual similarity.
            
            is_conf_junk = score >= conf_thresh
            is_conf_cell = score <= (1.0 - conf_thresh)
            
            if is_conf_junk or is_conf_cell:
                self.cluster_data[cid].append({
                    "h5_path": fp, 
                    "row_idx": ridx, 
                    "score": score,
                    "pred": "Junk" if is_conf_junk else "Cell"
                })
                
        self.current_cluster = 0
        self.refresh_gallery()
        QMessageBox.information(self, "Done", f"Analysis complete. Found items in {len(self.cluster_data)} clusters.")

    def change_cluster(self, delta):
        self.current_cluster = (self.current_cluster + delta) % self.spin_k.value()
        self.refresh_gallery()

    def refresh_gallery(self):
        items = self.cluster_data.get(self.current_cluster, [])
        self.to_drop.clear() # Reset drop selection on new cluster view
        self.lbl_cluster.setText(f"Cluster {self.current_cluster} ({len(items)} items)")
        
        # Render tiles
        tiles = []
        # Optimization: Group by file
        by_file = {}
        for it in items[:100]: # Limit display to 100 for speed
            by_file.setdefault(it["h5_path"], []).append(it["row_idx"])
            
        rgb_map = {}
        for fp, rows in by_file.items():
            imgs = read_images_by_indices(fp, np.array(rows), image_key=self.cfg.image_key)
            rgb_map[fp] = dict(zip(rows, [channels_to_rgb8bit(im) for im in imgs]))
            
        for it in items[:100]:
            fp, r = it["h5_path"], it["row_idx"]
            tiles.append({
                "h5_path": fp, "row_idx": r,
                "rgb": rgb_map[fp][r],
                "label": f"{it['pred']} ({it['score']:.2f})",
                "tooltip": f"{fp} {r}"
            })
            
        self.gallery.set_tiles(tiles)
        self.gallery.set_layout(8, 84, 84)

    def on_tile_clicked(self, fp, r, b):
        # Toggle drop status
        key = (fp, r)
        if key in self.to_drop:
            self.to_drop.remove(key)
            self.gallery.set_tile_label(fp, r, "Keep") # Visual feedback needed?
        else:
            self.to_drop.add(key)
            self.gallery.set_tile_label(fp, r, "DROP")

    def bulk_label(self, label_val):
        # Label all items in current cluster EXCEPT to_drop
        items = self.cluster_data.get(self.current_cluster, [])
        valid_items = [it for it in items if (it["h5_path"], it["row_idx"]) not in self.to_drop]
        
        if not valid_items: return
        
        # Write to HDF5
        by_file = {}
        for it in valid_items:
            by_file.setdefault(it["h5_path"], []).append(it["row_idx"])
            
        count = 0
        for fp, rows in by_file.items():
            try:
                write_features_rows_inplace(fp, rows, self.annotate_tab.label_col, [label_val]*len(rows), features_key=self.cfg.features_key)
                count += len(rows)
            except Exception as e: print(f"Write error: {e}")
            
        QMessageBox.information(self, "Labeled", f"Labeled {count} items as {label_val}.")
        
        # Remove these from the cluster list so they don't show up again
        new_list = [it for it in items if (it["h5_path"], it["row_idx"]) in self.to_drop]
        self.cluster_data[self.current_cluster] = new_list
        self.refresh_gallery()