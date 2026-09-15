from __future__ import annotations

import torch
from torch import nn


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")
        self.head_size = head_size
        self.rotary_dim = rotary_dim

        # inv_freq[i] = base ^ (-2i / rotary_dim)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        # 缓存 [max_pos, rotary_dim] = concat(cos, sin)，各占一半宽度
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos(), freqs.sin()), dim=-1),
            persistent=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = positions.reshape(-1)
        cos_sin = self.cos_sin_cache.index_select(0, flat).to(dtype=query.dtype)
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        def _apply(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            x = x.reshape(flat.numel(), -1, self.head_size)
            rot = _rotate(x[..., : self.rotary_dim], cos, sin)
            if self.rotary_dim < self.head_size:
                rot = torch.cat((rot, x[..., self.rotary_dim :]), dim=-1)
            return rot.reshape(shape)

        return _apply(query), _apply(key)
