# app/data/h5io.py
import h5py
import numpy as np
import pandas as pd

def list_feature_columns(h5_path: str, features_key: str = "features") -> list:
    try:
        with pd.HDFStore(h5_path, "r") as store:
            st = store.get_storer(features_key)
            if st and st.non_index_axes:
                return [str(c) for c in st.non_index_axes[0][1]]
            return []
    except Exception:
        return []

def read_features_columns(h5_path: str, cols: list, features_key: str = "features") -> pd.DataFrame:
    if not cols: return pd.DataFrame()
    return pd.read_hdf(h5_path, key=features_key, columns=cols)

def write_features_rows_inplace(h5_path: str, indices: list, col_name: str, values: list, features_key: str = "features"):
    if len(indices) == 0: return
    # This is a slow operation in HDFStore if not carefully managed. 
    # For safety/simplicity in this app, we read-update-write if small, or use h5py if simple array.
    # Assuming features is a DataFrame stored in 'table' format.
    df = pd.read_hdf(h5_path, key=features_key)
    if col_name not in df.columns:
        df[col_name] = float('nan')
    
    # Update
    df.loc[df.index[indices], col_name] = values
    
    # Write back (PyTables)
    df.to_hdf(h5_path, key=features_key, mode='a', format='table', data_columns=True)

def write_features_column_inplace(h5_path: str, col_name: str, values: np.ndarray, features_key: str = "features"):
    """Overwrites or creates an entire column."""
    df = pd.read_hdf(h5_path, key=features_key)
    df[col_name] = values
    df.to_hdf(h5_path, key=features_key, mode='a', format='table', data_columns=True)

def read_images_by_indices(h5_path: str, sorted_indices: np.ndarray, image_key: str = "images") -> np.ndarray:
    """
    Bulk read optimization.
    sorted_indices must be sorted ascending.
    """
    if len(sorted_indices) == 0: return np.array([])
    
    with h5py.File(h5_path, 'r') as f:
        # Fancy indexing in h5py supports passing a list/array of indices
        # provided they are increasing.
        return f[image_key][sorted_indices]