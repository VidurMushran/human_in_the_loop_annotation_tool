from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from .h5io import read_features_columns

@dataclass
class RowRef:
    h5_path: str
    row_idx: int

class MultiFileIndex:
    """
    Maintains a merged list of row references across selected HDF5 files.
    Stores only the feature columns you request (e.g., sort + label + metadata).
    """
    def __init__(self):
        self.refs: List[RowRef] = []
        self.table: pd.DataFrame = pd.DataFrame()  # aligned with refs

    def build(
        self,
        h5_paths: List[str],
        needed_cols: List[str],
        features_key: str,
    ):
        refs = []
        parts = []
        for p in h5_paths:
            df = read_features_columns(p, needed_cols, features_key=features_key).copy()
            df.reset_index(drop=True, inplace=True)
            n = len(df)
            refs.extend([RowRef(p, i) for i in range(n)])
            df["__h5_path__"] = p
            df["__row_idx__"] = np.arange(n, dtype=int)
            parts.append(df)

        self.refs = refs
        self.table = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()

    def sort(self, col: str, ascending: bool):
        if col not in self.table.columns:
            return
        self.table.sort_values(col, ascending=ascending, inplace=True, kind="mergesort")
        self.table.reset_index(drop=True, inplace=True)

    def sort_hard_cases(self, score_col: str):
        """
        Sort by abs(score - 0.5) ascending (hardest / most ambiguous first).
        """
        df = self.df  # whatever you store internally
        s = df[score_col].astype(float)
        df["__hard__"] = (s - 0.5).abs()
        df.sort_values("__hard__", ascending=True, inplace=True, kind="mergesort")
        df.drop(columns=["__hard__"], inplace=True)

    def page(self, start: int, count: int) -> pd.DataFrame:
        return self.table.iloc[start:start+count].copy()

    def size(self) -> int:
        return 0 if self.table is None else len(self.table)
