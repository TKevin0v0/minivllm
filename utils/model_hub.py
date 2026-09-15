from __future__ import annotations
import os
from pathlib import Path
import torch
from dataclasses import dataclass
from huggingface_hub import snapshot_download
__all__ = [
    "ModelSpec",
    "QWEN3_REGISTRY",
    "DEFAULT_CACHE_DIR",
    "is_local_model",
    "resolve_model_path",
    "download_model",
    "download_models",
    "is_downloaded",
    "gpu_memory_gb",
    "select_models_for_vram",
]

DEFAULT_CACHE_DIR = Path.home() / "huggingface"
@dataclass(frozen=True)
class ModelSpec:
    local_name: str
    repo_id: str
    fp16_gb: float
QWEN3_REGISTRY = (ModelSpec("Qwen3-0.6B", "Qwen/Qwen3-0.6B", 1.2), ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B", 3.4))
def is_local_model(path: Path) -> bool:
    return path.is_dir() and ((path / "config.json").is_file() or any(path.glob("*.safetensors")))
def resolve_model_path(model):
    path = Path(model).expanduser()
    return path if is_local_model(path) else DEFAULT_CACHE_DIR / path.name
def download_model(spec, cache_dir=None):
    target = (cache_dir or DEFAULT_CACHE_DIR) / spec.local_name
    if not is_local_model(target):
        snapshot_download(spec.repo_id, local_dir=str(target), token=os.environ.get("HF_TOKEN"))
    return target
def download_models(specs=None, cache_dir=None):
    return [download_model(s, cache_dir) for s in (specs or QWEN3_REGISTRY)]
def is_downloaded(spec, cache_dir=None):
    return is_local_model((cache_dir or DEFAULT_CACHE_DIR) / spec.local_name)
def gpu_memory_gb():
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return npu.get_device_properties(0).total_memory / 1024**3
    return torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0.0
def select_models_for_vram(vram_gb=None, reserve_gb=4.0, registry=QWEN3_REGISTRY):
    budget = (vram_gb or gpu_memory_gb() or 24.0) - reserve_gb
    return [s for s in registry if s.fp16_gb <= budget]
