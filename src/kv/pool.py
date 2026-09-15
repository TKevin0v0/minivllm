from __future__ import annotations
from dataclasses import dataclass, field
import torch
from torch import nn


@dataclass
class _Block:
    ref_count: int = 0
    hash: int = -1
    token_ids: list[int] = field(default_factory=list)

    def update(self, h: int, token_ids: list[int]) -> None:
        self.hash = h
        self.token_ids = list(token_ids)

    def clear_meta(self) -> None:
        self.hash = -1
        self.token_ids = []


class BlockPool:
    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.blocks = [_Block() for _ in range(num_blocks)]
        self.free_ids: set[int] = set(range(num_blocks))
        self.used_ids: set[int] = set()
        self.cached_ids: set[int] = set()
        self.on_evict = None  # optional Callable[[int], None]
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    @classmethod
    def allocate(
        cls,
        model: nn.Module,
        *,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BlockPool:
        layer0 = model.model.layers[0].self_attn
        return cls(
            num_layers=len(model.model.layers),
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=layer0.num_kv_heads,
            head_dim=layer0.head_dim,
            device=device,
            dtype=dtype,
        )

    def allocate_fresh(self) -> int:
        """ 取一页空闲块；free 空则淘汰一块 cached """
        if not self.free_ids:
            if not self.cached_ids:
                raise RuntimeError("KV block pool exhausted")
            bid = next(iter(self.cached_ids))
            self._evict_cached(bid)
        bid = self.free_ids.pop()
        block = self.blocks[bid]
        block.ref_count = 1
        block.clear_meta()
        self.used_ids.add(bid)
        return bid

    def _evict_cached(self, bid: int) -> None:
        """ 丢掉一块 cached，必要时通知前缀索引 """
        if self.on_evict is not None:
            self.on_evict(bid)
        self.cached_ids.discard(bid)
        self.blocks[bid].ref_count = 0
        self.blocks[bid].clear_meta()
        self.free_ids.add(bid)

    def reclaim_cached(self, bid: int) -> None:
        """ 前缀命中：把 cached 页重新变为 used """
        self.cached_ids.discard(bid)
        self.free_ids.discard(bid)
        self.used_ids.add(bid)
        self.blocks[bid].ref_count = 1

    def release_to_cached_or_free(self, bid: int, *, keep_cached: bool) -> None:
        """ 请求结束减引用；可保留为 cached，否则回 free """
        block = self.blocks[bid]
        block.ref_count -= 1
        if block.ref_count > 0:
            return
        self.used_ids.discard(bid)
        if keep_cached and block.hash != -1:
            self.cached_ids.add(bid)
        else:
            block.clear_meta()
            self.cached_ids.discard(bid)
            self.free_ids.add(bid)
