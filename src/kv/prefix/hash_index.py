from __future__ import annotations

from typing import TYPE_CHECKING

from .hash_util import compute_block_hash

if TYPE_CHECKING:
    from ..pool import BlockPool
    from ..sequence import Sequence


class HashPrefixIndex:
    def __init__(self) -> None:
        self.hash_to_block_id: dict[int, int] = {}

    def match(self, seq: Sequence, pool: BlockPool) -> list[int]:
        h = -1
        hit: list[int] = []
        for i in range(seq.num_full_blocks()):
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            bid = self.hash_to_block_id.get(h, -1)
            if bid < 0:
                break
            block = pool.blocks[bid]
            if block.token_ids != toks:
                break
            if bid not in pool.used_ids and bid not in pool.cached_ids:
                break
            hit.append(bid)
        return hit

    def publish(self, seq: Sequence, num_full_blocks: int, pool: BlockPool) -> None:
        if num_full_blocks <= 0:
            return
        h = -1
        for i in range(num_full_blocks):
            bid = seq.block_table[i]
            toks = seq.block_token_ids(i)
            h = compute_block_hash(toks, h)
            pool.blocks[bid].update(h, toks)
            old = self.hash_to_block_id.get(h, -1)
            if old < 0 or old == bid or pool.blocks[old].token_ids == toks:
                self.hash_to_block_id[h] = bid

    def invalidate(self, block_id: int, pool: BlockPool) -> None:
        block = pool.blocks[block_id]
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
