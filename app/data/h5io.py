from __future__ import annotations

from typing import List, Sequence
import numpy as np
import pandas as pd
import h5py
import logging
import warnings
from functools import lru_cache

log = logging.getLogger("h5io")


@lru_cache(maxsize=32)
def _open_h5(path: str) -> h5py.File:
    # IMPORTANT: open read-only and reuse. If you multi-thread, open per thread instead.
    return h5py.File(path, "r")


def list_feature_columns(h5_path: str, features_key: str = "features") -> List[str]:
    # 1. Fast path: Direct h5py read
    try:
        with h5py.File(h5_path, "r") as f:
            if features_key in f:
                grp = f[features_key]
                keys = list(grp.keys())
                # If it's not a pandas internal block format, return keys directly
                if "block0_values" not in keys and "table" not in keys:
                    return keys
    except Exception:
        pass

    # 2. Fallback to pandas with PyTables warnings explicitly suppressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = pd.read_hdf(h5_path, key=features_key, stop=1)
            return list(df.columns)
        except Exception:
            try:
                df = pd.read_hdf(h5_path, key=features_key)
                return list(df.columns)
            except Exception:
                return []


def read_features_columns(
    h5_path: str,
    columns: List[str],
    features_key: str = "features",
) -> pd.DataFrame:
    # 1. Fast path: Direct h5py read (bypasses pandas entirely)
    try:
        with h5py.File(h5_path, "r") as f:
            if features_key in f:
                grp = f[features_key]
                keys = list(grp.keys())
                
                if "block0_values" not in keys and "table" not in keys:
                    out_dict = {}
                    row_count = 0
                    for k in keys:
                        if hasattr(grp[k], "shape") and len(grp[k].shape) > 0:
                            row_count = grp[k].shape[0]
                            break
                            
                    for c in columns:
                        if c in grp:
                            out_dict[c] = np.array(grp[c])
                        else:
                            out_dict[c] = np.full(row_count, np.nan)
                            
                    return pd.DataFrame(out_dict)
    except Exception:
        pass

    # 2. Fallback to Pandas (with warnings strictly ignored)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with pd.HDFStore(h5_path, mode="r") as store:
                st = store.get_storer(features_key)
                if st is not None and getattr(st, "is_table", False):
                    df = store.select(features_key, columns=columns)
                    missing = [c for c in columns if c not in df.columns]
                    for c in missing:
                        df[c] = np.nan
                    return df[columns]
        except Exception:
            pass

        try:
            df = pd.read_hdf(h5_path, key=features_key)
            missing = [c for c in columns if c not in df.columns]
            for c in missing:
                df[c] = np.nan
            return df[columns]
        except Exception as e:
            log.error(f"[h5io] All read methods failed for {h5_path}: {e}")
            return pd.DataFrame({c: [] for c in columns})


def read_images_by_indices(h5_path: str, indices: np.ndarray, image_key: str = "images") -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty((0,), dtype=np.uint8)

    order = np.argsort(indices)
    sorted_idx = indices[order]

    with h5py.File(h5_path, "r") as f:
        X = f[image_key]

        blocks = []
        starts = [int(sorted_idx[0])]
        ends = []

        for a, b in zip(sorted_idx[:-1], sorted_idx[1:]):
            if int(b) != int(a) + 1:
                ends.append(int(a) + 1)
                starts.append(int(b))
        ends.append(int(sorted_idx[-1]) + 1)

        for s, e in zip(starts, ends):
            blocks.append(X[s:e])

        out_sorted = np.concatenate(blocks, axis=0)

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return out_sorted[inv]


def _delete_node_if_exists_h5py(h5_path: str, key: str):
    try:
        with h5py.File(h5_path, "a") as f:
            if key in f:
                del f[key]
    except Exception:
        pass


def _safe_put_features(store: pd.HDFStore, key: str, df: pd.DataFrame):
    try:
        store.put(key, df, format="table", data_columns=True, index=False)
    except Exception as e:
        log.warning(f"[h5io] store.put(table) failed for {key}: {e} ; falling back to fixed")
        store.put(key, df, format="fixed")


def write_features_column_inplace(
    h5_path: str,
    col_name: str,
    values: Sequence,
    features_key: str = "features",
) -> None:
    # 1. Fast path: Direct h5py modification
    try:
        with h5py.File(h5_path, "a") as f:
            if features_key in f:
                grp = f[features_key]
                keys = list(grp.keys())
                if "block0_values" not in keys and "table" not in keys:
                    values_arr = np.asarray(values)
                    if col_name in grp:
                        del grp[col_name]
                    grp.create_dataset(col_name, data=values_arr)
                    return
    except Exception as e:
        log.warning(f"[h5io] Direct h5py write failed: {e}")

    # 2. Fallback to pandas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = pd.read_hdf(h5_path, key=features_key)
            if len(values) != len(df):
                raise ValueError(f"Length mismatch: features={len(df)} values={len(values)}")

            df[col_name] = values
            _delete_node_if_exists_h5py(h5_path, features_key)

            with pd.HDFStore(h5_path, mode="a") as store:
                _safe_put_features(store, features_key, df)
            return
        except Exception as e:
            log.warning(f"[h5io] Pandas fallback write failed for {h5_path}: {e}")

    # 3. Absolute fallback: Force h5py raw dataset write (fixes the crash!)
    try:
        with h5py.File(h5_path, "a") as f:
            if features_key not in f:
                f.create_group(features_key)
            grp = f[features_key]
            values_arr = np.asarray(values)
            if col_name in grp:
                del grp[col_name]
            grp.create_dataset(col_name, data=values_arr)
    except Exception as e:
        log.error(f"[h5io] Absolute fallback failed for {h5_path}: {e}")
        raise RuntimeError(f"Could not write to {h5_path} using any method.") from e


def write_features_rows_inplace(
    h5_path: str,
    row_indices: np.ndarray,
    col_name: str,
    col_values: Sequence,
    features_key: str = "features",
) -> None:
    row_indices = np.asarray(row_indices, dtype=int)
    col_values = np.asarray(col_values)

    # 1. Fast path: Direct h5py modification
    try:
        with h5py.File(h5_path, "a") as f:
            if features_key in f:
                grp = f[features_key]
                keys = list(grp.keys())
                if "block0_values" not in keys and "table" not in keys:
                    if col_name not in grp:
                        row_count = 0
                        for k in keys:
                            if hasattr(grp[k], "shape") and len(grp[k].shape) > 0:
                                row_count = grp[k].shape[0]
                                break
                        grp.create_dataset(col_name, data=np.full(row_count, np.nan))
                    
                    data = np.array(grp[col_name])
                    data[row_indices] = col_values
                    del grp[col_name]
                    grp.create_dataset(col_name, data=data)
                    return
    except Exception as e:
        log.warning(f"[h5io] Direct h5py row write failed: {e}")

    # 2. Fallback to pandas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = pd.read_hdf(h5_path, key=features_key)
            if len(row_indices) != len(col_values):
                raise ValueError("row_indices and col_values must match length")

            df.loc[df.index[row_indices], col_name] = list(col_values)
            _delete_node_if_exists_h5py(h5_path, features_key)

            with pd.HDFStore(h5_path, mode="a") as store:
                _safe_put_features(store, features_key, df)
            return
        except Exception as e:
            log.warning(f"[h5io] Pandas fallback row write failed for {h5_path}: {e}")

    # 3. Absolute fallback
    try:
        with h5py.File(h5_path, "a") as f:
            if features_key not in f:
                f.create_group(features_key)
            grp = f[features_key]
            if col_name not in grp:
                row_count = max([grp[k].shape[0] for k in grp.keys() if hasattr(grp[k], "shape") and len(grp[k].shape)>0] + [0])
                grp.create_dataset(col_name, data=np.full(row_count, np.nan))
            data = np.array(grp[col_name])
            data[row_indices] = col_values
            del grp[col_name]
            grp.create_dataset(col_name, data=data)
    except Exception as e:
        log.error(f"[h5io] Absolute fallback failed for {h5_path}: {e}")
        raise RuntimeError(f"Could not write rows to {h5_path} using any method.") from e