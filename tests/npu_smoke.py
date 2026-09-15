"""Small Ascend smoke test; run inside the server's torch_npu environment."""
from __future__ import annotations

import torch
import torch_npu  # noqa: F401  # registers the NPU backend

from src.device import resolve_device


def main() -> None:
    device = resolve_device("npu")
    x = torch.randn(1, 2, 8, 16, device=device, dtype=torch.float16)
    y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True)
    assert y.shape == x.shape
    cache = torch.zeros(32, 16, device=device, dtype=torch.float16)
    cache.index_copy_(0, torch.tensor([1, 2], device=device), torch.ones(2, 16, device=device, dtype=cache.dtype))
    torch.npu.synchronize()
    print(f"npu_smoke_ok device={device} sdpa={tuple(y.shape)} cache={float(cache.sum())}")


if __name__ == "__main__":
    main()
