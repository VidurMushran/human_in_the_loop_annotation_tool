from __future__ import annotations

from typing import List, Sequence
import numpy as np
import pandas as pd
import h5py
import logging
from functools import lru_cache

log = logging.getLogger("h5io")


@lru_cache(maxsize=32)
def _open_h5(path: str) -> h5py.File:
    # IMPORTANT: open read-only and reuse. If you multi-thread, open per thread instead.
    return h5py.File(path, "r")


def list_feature_columns(h5_path: str, features_key: str = "features") -> List[str]:
    try:
        df = pd.read_hdf(h5_path, key=features_key, stop=1)
        return list(df.columns)
    except Exception:
        df = pd.read_hdf(h5_path, key=features_key)
        return list(df.columns)


def read_features_columns(
    h5_path: str,
    columns: List[str],
    features_key: str = "features",
) -> pd.DataFrame:
    """
    Read only selected columns from the features table when possible.
    Falls back to full read if the store isn't a table or select fails.

    This function is required by app/data/index.py (MultiFileIndex).
    """
    # Fast path: HDFStore select when stored as a table
    try:
        with pd.HDFStore(h5_path, mode="r") as store:
            st = store.get_storer(features_key)
            if st is not None and getattr(st, "is_table", False):
                df = store.select(features_key, columns=columns)
                # ensure missing columns are present
                missing = [c for c in columns if c not in df.columns]
                for c in missing:
                    df[c] = np.nan
                return df[columns]
    except Exception:
        pass

    # Fallback: read full features then subset
    df = pd.read_hdf(h5_path, key=features_key)
    missing = [c for c in columns if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    return df[columns]


def read_images_by_indices(h5_path: str, indices: np.ndarray, image_key: str = "images") -> np.ndarray:
    """
    Efficiently read requested rows from /images:
      - sort indices
      - read contiguous runs via slicing
      - re-order back to original request order
    """
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
    """
    Hard-delete /key if it exists as a group/dataset (covers the pandas 'Group' case),
    so pd.HDFStore can recreate it cleanly.
    """
    try:
        with h5py.File(h5_path, "a") as f:
            if key in f:
                del f[key]
    except Exception:
        pass


def _safe_put_features(store: pd.HDFStore, key: str, df: pd.DataFrame):
    """
    Try table write first (fast column selects), fallback to fixed if table fails.
    """
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
    df = pd.read_hdf(h5_path, key=features_key)
    if len(values) != len(df):
        raise ValueError(f"Length mismatch: features={len(df)} values={len(values)}")

    df[col_name] = values

    # If /features exists as a raw HDF5 group (not a pandas table), delete it first.
    _delete_node_if_exists_h5py(h5_path, features_key)

    with pd.HDFStore(h5_path, mode="a") as store:
        try:
            if f"/{features_key}" in store:
                store.remove(features_key)
        except Exception:
            _delete_node_if_exists_h5py(h5_path, features_key)

        _safe_put_features(store, features_key, df)


def write_features_rows_inplace(
    h5_path: str,
    row_indices: np.ndarray,
    col_name: str,
    col_values: Sequence,
    features_key: str = "features",
) -> None:
    df = pd.read_hdf(h5_path, key=features_key)

    row_indices = np.asarray(row_indices, dtype=int)
    if len(row_indices) != len(col_values):
        raise ValueError("row_indices and col_values must match length")

    df.loc[df.index[row_indices], col_name] = list(col_values)

    # If /features exists as raw group, delete it first.
    _delete_node_if_exists_h5py(h5_path, features_key)

    with pd.HDFStore(h5_path, mode="a") as store:
        try:
            if f"/{features_key}" in store:
                store.remove(features_key)
        except Exception:
            _delete_node_if_exists_h5py(h5_path, features_key)

        _safe_put_features(store, features_key, df)
