# app/data/h5io.py
import h5py
import numpy as np
import pandas as pd
import warnings
from tables import NaturalNameWarning

# Suppress the specific warning about object names (harmless but annoying)
warnings.filterwarnings('ignore', category=NaturalNameWarning)

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Removes 'Unnamed' columns and resets index if needed."""
    # Drop artifacts from CSV roundtrips
    cols_to_drop = [c for c in df.columns if "Unnamed" in str(c)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df

def list_feature_columns(h5_path: str, features_key: str = "features") -> list:
    try:
        # Fast check with h5py first
        with h5py.File(h5_path, 'r') as f:
            if features_key not in f: return []
        
        with pd.HDFStore(h5_path, "r") as store:
            if features_key not in store: return []
            st = store.get_storer(features_key)
            if st and st.non_index_axes:
                return [str(c) for c in st.non_index_axes[0][1]]
            return []
    except Exception:
        return []

def read_features_columns(h5_path: str, cols: list, features_key: str = "features") -> pd.DataFrame:
    """Robust reader that attempts recovery if metadata is corrupt."""
    if not cols: return pd.DataFrame()
    
    try:
        # Standard fast read
        df = pd.read_hdf(h5_path, key=features_key, columns=cols)
        return _clean_dataframe(df)
        
    except (TypeError, ValueError, KeyError):
        # Fallback: Metadata corruption (your specific error)
        # Try reading the raw column nodes using h5py
        try:
            data = {}
            with h5py.File(h5_path, 'r') as f:
                if features_key not in f: return pd.DataFrame()
                grp = f[features_key]
                
                # Check for 'table' format (PyTables standard)
                if 'table' in grp:
                    # It's a structured array
                    import tables
                    with tables.open_file(h5_path, 'r') as tf:
                        table = tf.get_node(f"/{features_key}/table")
                        # Read only requested cols if possible, else all
                        arr = table.read()
                        # Convert to DF
                        full_df = pd.DataFrame(arr)
                        # Decode bytes to strings if needed
                        for c in full_df.select_dtypes([object]).columns:
                            full_df[c] = full_df[c].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
                        
                        # Filter cols
                        available = [c for c in cols if c in full_df.columns]
                        return _clean_dataframe(full_df[available])
                        
            return pd.DataFrame()
        except Exception as e:
            print(f"Deep recovery failed for {h5_path}: {e}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Read error {h5_path}: {e}")
        return pd.DataFrame()

def write_features_rows_inplace(h5_path: str, indices: list, col_name: str, values: list, features_key: str = "features"):
    if len(indices) == 0: return
    try:
        df = pd.read_hdf(h5_path, key=features_key)
        df = _clean_dataframe(df) # Clean before modifying
        
        if col_name not in df.columns:
            df[col_name] = float('nan')
        
        # Ensure indices line up (pandas indices vs row numbers)
        # We assume 'indices' passed in are integer row positions (0..N)
        # If DF has a custom index, we must use iloc-like logic or align index
        # For safety in this app, we assume default RangeIndex. 
        # If not, we reset it.
        if not isinstance(df.index, pd.RangeIndex):
            df.reset_index(drop=True, inplace=True)
            
        df.loc[indices, col_name] = values
        
        # Write back cleanly
        df.to_hdf(h5_path, key=features_key, mode='a', format='table', data_columns=True, complevel=5, complib='blosc')
        
    except Exception as e:
        print(f"Write error {h5_path}: {e}")

def write_features_column_inplace(h5_path: str, col_name: str, values: np.ndarray, features_key: str = "features"):
    try:
        df = pd.read_hdf(h5_path, key=features_key)
        df = _clean_dataframe(df)
        
        if len(values) != len(df):
            # Try to fix mismatched lengths (rare edge case)
            print(f"Length mismatch in {h5_path}. DF={len(df)}, Val={len(values)}")
            return

        df[col_name] = values
        
        df.reset_index(drop=True, inplace=True)
        
        df.to_hdf(h5_path, key=features_key, mode='a', format='table', data_columns=True, complevel=5, complib='blosc')
    except Exception as e:
        print(f"Write column error {h5_path}: {e}")

def read_images_by_indices(h5_path: str, sorted_indices: np.ndarray, image_key: str = "images") -> np.ndarray:
    if len(sorted_indices) == 0: return np.array([])
    try:
        with h5py.File(h5_path, 'r') as f:
            return f[image_key][sorted_indices]
    except Exception as e:
        print(f"Error reading images from {h5_path}: {e}")
        return np.array([])