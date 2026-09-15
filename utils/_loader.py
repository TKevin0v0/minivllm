from __future__ import annotations

import importlib


def export_module(globals_dict: dict, shim_file, name: str, *, depth: int) -> None:
    mod = importlib.import_module(f"utils.{name}")
    names = list(mod.__all__)
    globals_dict.update({n: getattr(mod, n) for n in names})
    globals_dict["__all__"] = names
