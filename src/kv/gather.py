from __future__ import annotations

import torch

from ..pagedattention import PagedStore, slots_tensor
from .pool import BlockPool
from .sequence import Sequence


def _write_kv(
    pool: BlockPool,
    seq: Sequence,
    layer_id: int,
    start: int,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    n = k.shape[1]
    store = PagedStore(pool)
    slots = slots_tensor(
        seq.block_table,
        start,
        start + n,
        pool.block_size,
        device=pool.k.device,
    )
    store.write(layer_id, slots, k, v)


def _gather_kv(
    pool: BlockPool,
    seq: Sequence,
    layer_id: int,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return PagedStore(pool).gather(layer_id, seq.block_table, length)
