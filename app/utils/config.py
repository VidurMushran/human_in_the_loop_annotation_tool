from dataclasses import dataclass, asdict
import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".junk_gui_config.yml"

@dataclass
class AppConfig:
    image_key: str = "images"
    features_key: str = "features"
    default_label_col: str = "label"
    default_score_col: str = "model_score"
    tile_px: int = 96
    page_size: int = 120
    n_cols: int = 8
    export_cells_dirname: str = "vidur_cells"
    export_junk_dirname: str = "vidur_junk"
    runs_dir: str = "runs"

def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if path.exists():
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        return AppConfig(**{**asdict(AppConfig()), **d})
    return AppConfig()

def save_config(cfg: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
