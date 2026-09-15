from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype

        x32 = x.float()
        if residual is not None:
            x32 = x32 + residual.float()
            residual = x32.to(orig_dtype)
        out = (
            x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        ).to(orig_dtype)
        return (out, residual) if residual is not None else out
