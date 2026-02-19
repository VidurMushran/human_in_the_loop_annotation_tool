import argparse
import json
import pandas as pd
from pathlib import Path

def build_index_from_h5(h5_path, label_col=None):
    """
    Build index list from HDF5 features table.
    """
    h5_path = Path(h5_path)
    index = []

    with pd.HDFStore(h5_path, mode="r") as store:
        if "features" not in store:
            raise RuntimeError(f"No 'features' table found in {h5_path}")

        df = store["features"]

        if "image_id" not in df.columns:
            raise RuntimeError("features table missing image_id column")

        for row_idx, row in df.iterrows():
            entry = {
                "h5_path": str(h5_path),
                "img_key": "images",
                "idx": int(row["image_id"]),
                "slide_id": row.get("slide_id", None),
                "features_row": int(row_idx),
            }

            if label_col and label_col in df.columns:
                if not pd.isna(row[label_col]):
                    entry["label"] = int(row[label_col])

            index.append(entry)

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    index = build_index_from_h5(args.h5, args.label_col)

    with open(args.out, "w") as f:
        json.dump(index, f)

    print(f"Saved index with {len(index)} entries to {args.out}")


if __name__ == "__main__":
    main()
