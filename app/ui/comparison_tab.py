# app/ui/comparison_tab.py
from __future__ import annotations
import os
import torch
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QMessageBox, QFileDialog, QSplitter)
from PyQt5.QtCore import Qt
from app.utils.model_helpers import load_model_from_checkpoint
from app.data.h5io import read_images_by_indices
from app.ui.widgets.gallery_pane import GalleryPane
from app.imaging.render import channels_to_rgb8bit

class ComparisonTab(QWidget):
    def __init__(self, cfg, annotate_tab):
        super().__init__()
        self.cfg = cfg
        self.annotate_tab = annotate_tab
        self.model_a = None
        self.model_b = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        
        row = QHBoxLayout()
        self.btn_load_a = QPushButton("Load Model A")
        self.btn_load_a.clicked.connect(lambda: self.load_model('A'))
        self.lbl_a = QLabel("Model A: None")
        
        self.btn_load_b = QPushButton("Load Model B")
        self.btn_load_b.clicked.connect(lambda: self.load_model('B'))
        self.lbl_b = QLabel("Model B: None")
        
        self.btn_run = QPushButton("Compare on Selected Files")
        self.btn_run.clicked.connect(self.run_comparison)
        
        row.addWidget(self.btn_load_a); row.addWidget(self.lbl_a)
        row.addWidget(self.btn_load_b); row.addWidget(self.lbl_b)
        row.addWidget(self.btn_run)
        lay.addLayout(row)
        
        split = QSplitter(Qt.Horizontal)
        cb = lambda p,r,b: None
        self.gal_a_junk = GalleryPane("A=Junk | B=Cell", cb)
        self.gal_a_cell = GalleryPane("A=Cell | B=Junk", cb)
        split.addWidget(self.gal_a_junk)
        split.addWidget(self.gal_a_cell)
        lay.addWidget(split)

    def load_model(self, tag):
        p, _ = QFileDialog.getOpenFileName(self, f"Load Model {tag}", "", "*.pt")
        if not p: return
        try:
            m, _, _ = load_model_from_checkpoint(p, device="cuda")
            if tag == 'A': 
                self.model_a = m
                self.lbl_a.setText(f"A: {os.path.basename(p)}")
            else: 
                self.model_b = m
                self.lbl_b.setText(f"B: {os.path.basename(p)}")
        except Exception as e: QMessageBox.warning(self, "Error", str(e))

    def run_comparison(self):
        if not (self.model_a and self.model_b): return QMessageBox.warning(self, "Error", "Load both models.")
        paths = self.annotate_tab.selected_paths
        if not paths: return QMessageBox.warning(self, "Error", "Select files.")
        
        disagree_ab = [] # A=Junk(1), B=Cell(0)
        disagree_ba = [] # A=Cell(0), B=Junk(1)
        
        all_items = []
        import h5py
        for fp in paths:
            try: 
                with h5py.File(fp,'r') as f: n = f[self.cfg.image_key].shape[0]
                all_items.extend([(fp, i) for i in range(n)])
            except: pass
            
        rng = np.random.default_rng(0)
        if len(all_items) > 1000:
            picks_idx = rng.choice(len(all_items), 1000, replace=False)
            all_items = [all_items[i] for i in picks_idx]
            
        by_file = {}
        for fp, r in all_items: by_file.setdefault(fp, []).append(r)
        
        self.model_a.eval(); self.model_b.eval()
        
        with torch.no_grad():
            for fp, rows in by_file.items():
                imgs = read_images_by_indices(fp, np.array(rows), image_key=self.cfg.image_key)
                xb = torch.from_numpy(imgs).permute(0,3,1,2).float().cuda()
                
                la = self.model_a(xb); pa = torch.softmax(la, 1)[:,1].cpu().numpy()
                pred_a = (pa >= 0.5).astype(int)
                
                lb = self.model_b(xb); pb = torch.softmax(lb, 1)[:,1].cpu().numpy()
                pred_b = (pb >= 0.5).astype(int)
                
                rgb_imgs = [channels_to_rgb8bit(im) for im in imgs]
                
                for i, (ya, yb) in enumerate(zip(pred_a, pred_b)):
                    item = {"h5_path":fp, "row_idx":rows[i], "rgb":rgb_imgs[i], "label":"", "tooltip":f"{fp} {rows[i]}"}
                    if ya==1 and yb==0: disagree_ab.append(item)
                    elif ya==0 and yb==1: disagree_ba.append(item)
                    
        self.gal_a_junk.set_tiles(disagree_ab)
        self.gal_a_cell.set_tiles(disagree_ba)
        self.gal_a_junk.set_layout(6, 84, 84)
        self.gal_a_cell.set_layout(6, 84, 84)
        QMessageBox.information(self, "Done", f"Found {len(disagree_ab)} A=J/B=C and {len(disagree_ba)} A=C/B=J disagreements.")