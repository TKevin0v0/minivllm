from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..pagedattention import PagedStore
from .gather import _gather_kv, _write_kv
from .pool import BlockPool
from .prefix import HashPrefixIndex, NoopPrefixIndex, RadixPrefixIndex
from .sequence import Sequence

_PrefixBackend = Literal["none", "hash", "radix"]


def _make_index(backend: _PrefixBackend):
    if backend == "hash":
        return HashPrefixIndex()
    if backend == "radix":
        return RadixPrefixIndex()
    return NoopPrefixIndex()


class KVManager:
    def __init__(
        self,
        model: nn.Module,
        *,
        num_blocks: int,
        block_size: int,
        prefix_backend: _PrefixBackend,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.block_size = block_size
        self.prefix_backend = prefix_backend
        self.pool = BlockPool.allocate(
            model,
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )
        self.store = PagedStore(self.pool)
        self.index = _make_index(prefix_backend)
        self.pool.on_evict = lambda bid: self.index.invalidate(bid, self.pool)
        self.last_num_cached_tokens = 0

    def _fit_prefix(self, seq: Sequence) -> int:
        """前缀复用预检查：计算当前序列最多可复用多少连续前缀 KV 块。"""
        need = seq.num_blocks
        if need == 0:
            return 0
        hit = self.index.match(seq, self.pool)
        max_cached = min(len(hit), need)
        reclaimable = sum(1 for b in hit[:max_cached] if b in self.pool.cached_ids)
        other_cached = len(self.pool.cached_ids) - reclaimable
        new_needed = need - max_cached
        if len(self.pool.free_ids) + other_cached >= new_needed:
            return max_cached
        return -1

    def can_allocate(self, seq: Sequence) -> bool:
        """ 判断新请求是否有足够物理块可分配 """
        return self._fit_prefix(seq) >= 0

    def allocate(self, seq: Sequence) -> None:
        """ 匹配前缀，绑定已缓存的 KV 块，剩余部分分配全新物理块 """
        seq.block_table.clear()
        seq.num_cached_tokens = 0
        need = seq.num_blocks
        if need == 0:
            self.last_num_cached_tokens = 0
            return

        max_cached = self._fit_prefix(seq)
        if max_cached < 0:
            raise RuntimeError("not enough KV blocks for sequence")
        hit = self.index.match(seq, self.pool)
        for bid in hit[:max_cached]:
            if bid in self.pool.used_ids:
                self.pool.blocks[bid].ref_count += 1
            elif bid in self.pool.cached_ids:
                self.pool.reclaim_cached(bid)
            else:
                raise RuntimeError(f"prefix hit on non-resident block {bid}")
            seq.block_table.append(bid)

        for _ in range(max_cached, need):
            seq.block_table.append(self.pool.allocate_fresh())

        seq.num_cached_tokens = max_cached * self.block_size
        self.last_num_cached_tokens = seq.num_cached_tokens

        if isinstance(self.index, RadixPrefixIndex) and max_cached > 0:
            # 如果索引是基树索引且有复用前缀，需要更新基树引用
            # 同时给树上已存在的节点增加引用
            key = self.index._block_hashes(seq, max_cached)  # noqa: SLF001
            _ids, node = self.index.tree.match_prefix(key)
            self.index.tree.inc_ref(node)
            self.index._seq_node[seq.seq_id] = node  # noqa: SLF001

    def can_append(self, seq: Sequence) -> bool:
        """ Decode阶段判断当前能不能给这个 sequence append 1 个新 token """
        need_new = len(seq) % self.block_size == 1
        return (not need_new) or len(self.pool.free_ids) >= 1

    def may_append(self, seq: Sequence) -> None:
        """ Decode 生成阶段，执行真正的 block 分配动作 """
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self.pool.allocate_fresh())

    def ensure_blocks(self, seq: Sequence) -> None:
        """ Decode 变长时把 block_table 补到够用 """
        while len(seq.block_table) < seq.num_blocks:
            seq.block_table.append(self.pool.allocate_fresh())

    def sync_prefix(self, seq: Sequence) -> None:
        """ postprocess()调用 把这个序列已经完整填满的 KV block，发布到 prefix 索引 """
        n_full = seq.num_full_blocks()
        if n_full > 0 and self.prefix_backend != "none":
            self.index.publish(seq, n_full, self.pool)

    def write(
        self,
        layer_id: int,
        seq: Sequence,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        _write_kv(self.pool, seq, layer_id, start, k, v)

    def gather(
        self, layer_id: int, seq: Sequence, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _gather_kv(self.pool, seq, layer_id, length)

    def deallocate(self, seq: Sequence, *, publish: bool) -> None:
        """ 释放一个 sequence 持有的全部物理 block """
        """ publish=False 任务被抢占暂停，不发布prefix缓存，尽量保留共享缓存数据 """
        """ publish=True 请求彻底跑完，把完整block提交给prefix缓存 """
        n_full = seq.num_full_blocks() if publish else 0
        if publish and n_full > 0:
            self.index.publish(seq, n_full, self.pool)

        if isinstance(self.index, RadixPrefixIndex):
            self.index.release_seq(seq.seq_id)

        published = set(seq.block_table[:n_full]) if n_full > 0 else set()
        # 本次 publish 要保留进prefix 缓存的物理 block id 集合
        for bid in list(seq.block_table):
            keep = bid in published and self.prefix_backend != "none"
            if not keep and self.prefix_backend == "hash":
                # hash 索引下，这个block不保留做共享缓存，从hash索引作废
                self.index.invalidate(bid, self.pool)
            self.pool.release_to_cached_or_free(bid, keep_cached=keep)
            # 需要复用时 published不为空，keep为true时，该物理块进入cached缓存池，供其他seq复用
            # 需要释放显存时 published为空集合，keep为false，直接归还进free池，释放GPU资源

        seq.block_table.clear() # 清空请求自己的块表
        seq.num_cached_tokens = 0

    def finish(self, seq: Sequence) -> None:
        self.deallocate(seq, publish=True)
