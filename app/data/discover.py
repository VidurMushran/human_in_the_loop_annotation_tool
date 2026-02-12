from pathlib import Path
from typing import List, Tuple

SUBFOLDERS = [
    "junk_annotated",
    "rare_cells_annotated",
    "wbcs_annotated",
    "unannotated",
    "common_cell",
    "Common_Cell",
    "Dapi_",
    "Dapi__Cell",
    "Interesting",
    "Not_classified",
    "Not_interesting",
    "Not_sure",
    "Rare_Cell",
    "Unpacked",
    "Unsure",
]

def discover_hdf5s(root: str) -> List[Tuple[str, str]]:
    """
    Returns list of (group_name, file_path).
    """
    out = []
    rootp = Path(root)
    for sf in SUBFOLDERS:
        d = rootp / sf
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.hdf5")):
            out.append((sf, fp.as_posix()))
    return out

def group_to_default_class(group_name: str):
    # your convention:
    # junk_annotated => junk (1)
    # rare_cells_annotated + wbcs_annotated => cells (0)
    if group_name == "junk_annotated":
        return 1
    if group_name in ("rare_cells_annotated", "wbcs_annotated"):
        return 0
    return None
