from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..kv.context import get_forward_context


def _sdpa_supports_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 4)
        k = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=True)
        return True
    except TypeError:
        return False


_SDPA_HAS_GQA = _sdpa_supports_gqa()


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = scaling

    def _expand_gqa(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_kv_groups <= 1:
            return k, v
        b, hkv, s, d = k.shape
        k = k.unsqueeze(2).expand(b, hkv, self.num_kv_groups, s, d).reshape(
            b, self.num_heads, s, d
        )
        v = v.unsqueeze(2).expand(b, hkv, self.num_kv_groups, s, d).reshape(
            b, self.num_heads, s, d
        )
        return k, v

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict = {"scale": self.scaling}
        if attn_mask is not None:
            kwargs["attn_mask"] = attn_mask
        else:
            kwargs["is_causal"] = is_causal
        if _SDPA_HAS_GQA and k.shape[1] != q.shape[1]:
            kwargs["enable_gqa"] = True
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
        if k.shape[1] != q.shape[1]:
            k, v = self._expand_gqa(k, v)
        return F.scaled_dot_product_attention(q, k, v, **kwargs)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx = get_forward_context()
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if ctx is None:
            out = self._sdpa(q, k, v, is_causal=True)
            return out.transpose(1, 2).contiguous()

        store = ctx.store
        lid = self.layer_id
        store.write(lid, ctx.slot_mapping, k.transpose(1, 2), v.transpose(1, 2))

        if ctx.is_prefill:
            return self._prefill(q, k, v, ctx)
        return self._decode(q, ctx)

    def _prefill(self, q, k, v, ctx) -> torch.Tensor:
        cached = ctx.cached_lens[0] if ctx.cached_lens else 0
        ctx_len = ctx.context_lens[0]
        q_len = ctx.query_lens[0]

        if cached == 0:
            out = self._sdpa(q, k, v, is_causal=True)
            return out.transpose(1, 2).contiguous()

        k_full, v_full = ctx.store.gather(self.layer_id, ctx.block_tables[0], ctx_len)
        k_full = k_full.unsqueeze(0).transpose(1, 2)
        v_full = v_full.unsqueeze(0).transpose(1, 2)
        device = q.device
        qi = torch.arange(q_len, device=device).unsqueeze(1) + cached
        kj = torch.arange(ctx_len, device=device).unsqueeze(0)
        allow = qi >= kj
        attn_mask = torch.zeros(q_len, ctx_len, device=device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~allow, torch.finfo(q.dtype).min)
        out = self._sdpa(q, k_full, v_full, is_causal=False, attn_mask=attn_mask)
        return out.transpose(1, 2).contiguous()

    def _decode(self, q, ctx) -> torch.Tensor:
        outs = []
        for b in range(q.shape[0]):
            ctx_len = ctx.context_lens[b]
            k_c, v_c = ctx.store.gather(self.layer_id, ctx.block_tables[b], ctx_len)
            k_c = k_c.unsqueeze(0).transpose(1, 2)
            v_c = v_c.unsqueeze(0).transpose(1, 2)
            outs.append(self._sdpa(q[b : b + 1], k_c, v_c, is_causal=False))
        return torch.cat(outs, dim=0).transpose(1, 2).contiguous()
