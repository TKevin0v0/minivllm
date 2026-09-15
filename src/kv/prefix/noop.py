from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pool import BlockPool
    from ..sequence import Sequence


class NoopPrefixIndex:
    def match(self, seq: Sequence, pool: BlockPool) -> list[int]:
        return []

    def publish(self, seq: Sequence, num_full_blocks: int, pool: BlockPool) -> None:
        return

    def invalidate(self, block_id: int, pool: BlockPool) -> None:
        return
