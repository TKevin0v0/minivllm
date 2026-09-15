from __future__ import annotations

from typing import Protocol

import torch


class _BlockTensorStore(Protocol):
    # Protocol 协议类型，相当于接口定义,描述 KV 显存池对象必须具备的成员与张量 shape
    block_size: int
    num_kv_heads: int
    head_dim: int
    k: torch.Tensor  # [L, N_blocks, B, Hkv, D]
    v: torch.Tensor


def _coerce_kv_token_major(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """ 把 KV 统一转换成 token 主维度格式 [total_tokens, Hkv, D] """
    if k.dim() == 4:
        b, s, h, d = k.shape
        return k.reshape(b * s, h, d), v.reshape(b * s, h, d)
    if k.dim() == 3:
        return k, v
    raise ValueError(f"unexpected k rank {k.dim()}")


def _flat_layer_views(
    pool: _BlockTensorStore, layer_id: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """ 取某一层 layer_id 的全部 KV 池，做展平视图 """
    flat_dim = pool.num_kv_heads * pool.head_dim
    flat_k = pool.k[layer_id].reshape(-1, flat_dim)
    flat_v = pool.v[layer_id].reshape(-1, flat_dim)
    return flat_k, flat_v, flat_dim


class PagedStore:
    def __init__(self, pool: _BlockTensorStore) -> None:
        self.pool = pool

    @property
    def block_size(self) -> int:
        return self.pool.block_size

    @property
    def device(self) -> torch.device:
        return self.pool.k.device

    def write(
        self,
        layer_id: int,
        slot_mapping: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        k, v = _coerce_kv_token_major(k, v)
        n = k.shape[0]
        if slot_mapping.numel() != n:
            raise ValueError(
                f"slot_mapping length {slot_mapping.numel()} != seq length {n}"
            )
        flat_k, flat_v, flat_dim = _flat_layer_views(self.pool, layer_id)
        with torch.no_grad():
            flat_k.index_copy_(0, slot_mapping, k.reshape(n, flat_dim).detach())
            flat_v.index_copy_(0, slot_mapping, v.reshape(n, flat_dim).detach())

    def gather(
        self,
        layer_id: int,
        block_table: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            empty = self.pool.k.new_empty(0, self.pool.num_kv_heads, self.pool.head_dim)
            return empty, empty.clone()

        bs = self.pool.block_size
        need_blocks = (seq_len + bs - 1) // bs
        if len(block_table) < need_blocks:
            raise IndexError(
                f"seq_len={seq_len} needs {need_blocks} blocks, "
                f"block_table has {len(block_table)}"
            )

        device = self.pool.k.device
        table = torch.tensor(block_table[:need_blocks], device=device, dtype=torch.long)
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        physical = table[positions // bs]
        offsets = positions % bs
        k = self.pool.k[layer_id, physical, offsets]
        v = self.pool.v[layer_id, physical, offsets]
        return k, v
