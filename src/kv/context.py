from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..pagedattention import PagedStore


@dataclass
class ForwardContext:
    is_prefill: bool
    store: PagedStore
    block_tables: list[list[int]]
    slot_mapping: torch.Tensor
    query_lens: list[int]
    context_lens: list[int]
    cached_lens: list[int] = field(default_factory=list)


_forward_ctx: ForwardContext | None = None


def set_forward_context(ctx: ForwardContext | None) -> None:
    global _forward_ctx
    _forward_ctx = ctx


def get_forward_context() -> ForwardContext | None:
    return _forward_ctx
