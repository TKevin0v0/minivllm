from __future__ import annotations
from pathlib import Path
import yaml

__all__ = ["load_config", "merge_cli"]

def load_config(path: Path, defaults: dict) -> dict:
    cfg = dict(defaults)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg

def merge_cli(cfg: dict, args, keys) -> None:
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
