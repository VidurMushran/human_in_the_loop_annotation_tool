# app/data/batch_ops.py
import os
import numpy as np
import h5py
import pandas as pd
from app.data.h5io import write_features_column_inplace
from app.data.export import append_rows_to_h5

def batch_label_files(file_paths: list[str], label_col: str, label_val: float, image_key: str, features_key: str):
    """Writes a specific label value to all rows in the given files."""
    count = 0
    errors = []
    for fp in file_paths:
        try:
            with h5py.File(fp, "r") as f:
                n = f[image_key].shape[0]
            arr = np.full(n, label_val, dtype=np.float32)
            write_features_column_inplace(fp, label_col, arr, features_key=features_key)
            count += 1
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")
    return count, errors

def rebuild_vidur_files(file_paths: list[str], root_dir: str, label_col: str, 
                       junk_dirname: str, cells_dirname: str, 
                       image_key: str, features_key: str):
    """Scans files for label_col and exports 'junk' and 'cells' to consolidated files."""
    
    junk_dir = os.path.join(root_dir, junk_dirname)
    cell_dir = os.path.join(root_dir, cells_dirname)
    os.makedirs(junk_dir, exist_ok=True)
    os.makedirs(cell_dir, exist_ok=True)

    dst_junk = os.path.join(junk_dir, "vidur_junk.hdf5")
    dst_cell = os.path.join(cell_dir, "vidur_cells.hdf5")

    # Clean start
    for p in [dst_junk, dst_cell]:
        if os.path.exists(p):
            os.remove(p)

    n_junk = 0
    n_cell = 0

    for fp in file_paths:
        try:
            df = pd.read_hdf(fp, key=features_key)
        except Exception:
            continue
        
        if label_col not in df.columns:
            continue

        col = df[label_col]
        
        # Match numeric (1.0/0.0) or string ('junk'/'cell')
        if np.issubdtype(col.dtype, np.number):
            junk_mask = (col == 1)
            cell_mask = (col == 0)
        else:
            s = col.astype(str).str.lower()
            junk_mask = s.str.contains("junk") | (s == "1")
            cell_mask = s.str.contains("cell") | (s == "0")

        junk_idx = np.where(junk_mask.to_numpy())[0].astype(int)
        cell_idx = np.where(cell_mask.to_numpy())[0].astype(int)

        if len(junk_idx) > 0:
            append_rows_to_h5(fp, junk_idx, dst_junk, image_key=image_key, features_key=features_key)
            n_junk += len(junk_idx)

        if len(cell_idx) > 0:
            append_rows_to_h5(fp, cell_idx, dst_cell, image_key=image_key, features_key=features_key)
            n_cell += len(cell_idx)

    return dst_junk, n_junk, dst_cell, n_cell