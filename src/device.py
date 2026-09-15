"""Device/backend helpers shared by the v2 runtime."""
from __future__ import annotations
import importlib
import os
import torch

def resolve_device(preference: str = "auto") -> torch.device:
    preference = (os.environ.get("VLLM_DEVICE", preference) or "auto").lower()
    if preference not in {"auto", "npu", "cuda", "cpu"}:
        raise ValueError(f"device must be auto|npu|cuda|cpu, got {preference!r}")
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pass
    npu = getattr(torch, "npu", None)
    npu_available = bool(npu is not None and npu.is_available())
    cuda_available = bool(torch.cuda.is_available())
    if preference == "npu" or (preference == "auto" and npu_available):
        if not npu_available:
            raise RuntimeError("device=npu requested but torch_npu/NPU is unavailable")
        npu.set_device(0)
        return torch.device("npu:0")
    if preference == "cuda" or (preference == "auto" and cuda_available):
        if not cuda_available:
            raise RuntimeError("device=cuda requested but CUDA is unavailable")
        return torch.device("cuda:0")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")

def empty_cache(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
