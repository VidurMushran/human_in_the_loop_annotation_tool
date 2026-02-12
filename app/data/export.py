from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from app.data.h5io import read_images_by_indices

def _ensure_images_dataset(hf: h5py.File, image_key: str, img_shape, dtype):
    if image_key in hf:
        return hf[image_key]
    maxshape = (None,) + img_shape[1:]
    dset = hf.create_dataset(
        image_key,
        shape=(0,) + img_shape[1:],
        maxshape=maxshape,
        chunks=(min(256, img_shape[0]),) + img_shape[1:],
        dtype=dtype,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    return dset

def append_rows_to_h5(
    src_h5: str,
    row_indices: np.ndarray,
    dst_h5: str,
    image_key: str = "images",
    features_key: str = "features",
):
    row_indices = np.asarray(row_indices, dtype=int)
    if row_indices.size == 0:
        return

    # read features rows (full read; safe for initial version)
    df = pd.read_hdf(src_h5, key=features_key).iloc[row_indices].copy()
    df.reset_index(drop=True, inplace=True)
    df["__source_h5__"] = Path(src_h5).name

    imgs = read_images_by_indices(src_h5, row_indices, image_key=image_key)

    Path(dst_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst_h5, "a") as hf:
        dset = _ensure_images_dataset(hf, image_key, imgs.shape, imgs.dtype)
        n0 = dset.shape[0]
        dset.resize((n0 + imgs.shape[0],) + dset.shape[1:])
        dset[n0:n0 + imgs.shape[0]] = imgs

    with pd.HDFStore(dst_h5, mode="a") as store:
        store.append(features_key, df, format="table", data_columns=True, index=False)
