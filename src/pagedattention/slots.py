from __future__ import annotations

import torch


def _slot_of(block_table: list[int], pos: int, block_size: int) -> int:
    """ 单个逻辑 pos 换算成物理 slot """

    block_idx = pos // block_size
    if block_idx >= len(block_table):
        raise IndexError(
            f"pos={pos} needs block_idx={block_idx}, "
            f"but block_table has {len(block_table)} pages"
        )
    return block_table[block_idx] * block_size + (pos % block_size)


def _slots_for_range(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
) -> list[int]:
    """ 一段连续逻辑 pos 换算成物理 slot 列表 """
    if end < start:
        raise ValueError(f"end={end} < start={start}")
    return [_slot_of(block_table, pos, block_size) for pos in range(start, end)]


def slots_tensor(
    block_table: list[int],
    start: int,
    end: int,
    block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """ 把逻辑区间 [start, end) 内的每个位置换算成全局物理 slot。
    slot = 所在物理页号 * 页大小 + 页内偏移；页号由 block_table[pos // block_size] 给出。
    """
    slots = _slots_for_range(block_table, start, end, block_size)
    return torch.tensor(slots, device=device, dtype=torch.long)
