import argparse
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from app.data.lazy_h5_dataset import LazyH5Dataset
from app.models.model_factory import build_model
from app.training.uncertainty import entropy_from_logits, margin_from_logits


def score_model(index_json, model_path, batch_size=256, include_mask=False):
    with open(index_json) as f:
        index = json.load(f)

    dataset = LazyH5Dataset(index, include_mask=include_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_x, _ = dataset[0]
    in_ch = sample_x.shape[0]

    model = build_model(in_channels=in_ch)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []
    all_entropy = []
    all_margin = []
    all_img_ids = []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            ent = entropy_from_logits(logits)
            mar = margin_from_logits(logits)

            all_probs.append(probs.cpu())
            all_entropy.append(ent.cpu())
            all_margin.append(mar.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_entropy = torch.cat(all_entropy).numpy()
    all_margin = torch.cat(all_margin).numpy()

    return index, all_probs, all_entropy, all_margin


def write_back(index, probs, entropy, margin):
    # assumes all index entries refer to same H5
    h5_path = Path(index[0]["h5_path"])

    with pd.HDFStore(h5_path, mode="a") as store:
        df = store["features"]

        if "model_prob" not in df.columns:
            df["model_prob"] = np.nan
        if "model_entropy" not in df.columns:
            df["model_entropy"] = np.nan
        if "model_margin" not in df.columns:
            df["model_margin"] = np.nan

        for entry, p, e, m in zip(index, probs, entropy, margin):
            row_idx = entry["features_row"]
            df.at[row_idx, "model_prob"] = float(p[1])
            df.at[row_idx, "model_entropy"] = float(e)
            df.at[row_idx, "model_margin"] = float(m)

        store.remove("features")
        store.put("features", df, format="table", data_columns=True)

    print("Wrote scores to HDF5.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--include-mask", action="store_true")
    args = parser.parse_args()

    index, probs, entropy, margin = score_model(
        args.index,
        args.model,
        args.batch_size,
        args.include_mask
    )

    write_back(index, probs, entropy, margin)


if __name__ == "__main__":
    main()
