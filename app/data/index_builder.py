# app/data/index_builder.py
import pandas as pd
from pathlib import Path

def build_index_from_h5(h5_path, require_label=False, label_col="label"):
    """
    Returns list of metadata dicts for each image row in the HDFStore 'features' table.
    Each dict contains:
      - h5_path: Path to the .hdf5 file
      - image_id: int index into images dataset
      - img_key: 'images'
      - slide_id: slide id (if present)
      - label: optional integer label (if label_col present)
    """
    h5_path = Path(h5_path)
    idx_list = []
    with pd.HDFStore(h5_path, mode="r") as store:
        if "/features" not in store.keys() and "features" not in store.keys():
            raise RuntimeError(f"No 'features' table in {h5_path}")
        df = store["features"]
        # ensure image_id exists
        assert "image_id" in df.columns, "features table missing image_id column"
        for _, row in df.iterrows():
            entry = {
                "h5_path": str(h5_path),
                "img_key": "images",
                "idx": int(row["image_id"]),
                "slide_id": row.get("slide_id", None),
                "meta_row_index": int(row.name)  # optional pointer back to features table row index
            }
            if label_col in row and not pd.isna(row[label_col]):
                entry["label"] = int(row[label_col])
            elif require_label:
                continue
            idx_list.append(entry)
    return idx_list
