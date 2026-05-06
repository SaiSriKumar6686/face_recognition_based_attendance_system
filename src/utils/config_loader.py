"""
config_loader.py
────────────────
Load and validate the YAML master config.  All other modules import
cfg from here so there is a single source of truth.
"""

import os
from pathlib import Path
import yaml


_DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "config.yaml"


def load_config(path: str | Path = _DEFAULT_CONFIG) -> dict:
    """Load YAML config and resolve relative paths to absolute paths."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # resolve all path entries relative to project root
    project_root = path.parents[1]
    _resolve_paths(cfg.get("paths", {}), project_root)
    _resolve_path_key(cfg.get("database", {}), "path", project_root)
    _resolve_path_key(cfg.get("database", {}), "faiss_index", project_root)

    return cfg


def _resolve_paths(paths_dict: dict, root: Path) -> None:
    for key, val in paths_dict.items():
        if isinstance(val, str) and not os.path.isabs(val):
            paths_dict[key] = str(root / val)


def _resolve_path_key(d: dict, key: str, root: Path) -> None:
    if key in d and not os.path.isabs(d[key]):
        d[key] = str(root / d[key])


# module-level singleton — import with:  from src.utils.config_loader import cfg
cfg = load_config()


if __name__ == "__main__":
    import pprint
    pprint.pprint(cfg)
