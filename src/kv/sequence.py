from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

from ..sampling_params import SamplingParams

_SEQ_IDS = count(1)


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    token_ids: list[int]
    block_size: int
    seq_id: int = 0
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0 # 命中 KV 缓存的 token 数

    # v2新增
    status: SequenceStatus = SequenceStatus.WAITING
    num_scheduled_tokens: int = 0 # 本轮 step 要调度处理的 toke 数量
    is_prefill: bool = True
    num_prompt_tokens: int = 0 # prompt 输入部分 token 数量
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 128
    ignore_eos: bool = False
    last_token: int = 0

    @classmethod
    def from_prompt(
        cls,
        token_ids: list[int],
        *,
        block_size: int,
        sampling: SamplingParams | None = None,
    ) -> Sequence:
        """ 接收 token_ids 创建 Sequence 对象"""
        sp = sampling or SamplingParams()
        ids = copy(token_ids)
        if not ids:
            raise ValueError("empty prompt")
        return cls(
            token_ids=ids,
            block_size=block_size,
            seq_id=next(_SEQ_IDS), # 计数器，区分请求id
            num_prompt_tokens=len(ids),
            last_token=ids[-1],
            temperature=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            min_p=sp.min_p,
            max_tokens=sp.max_tokens,
            ignore_eos=sp.ignore_eos,
        )

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        # 输出 token 数
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_blocks(self) -> int:
        n = self.num_tokens
        if n == 0:
            return 0
        return (n + self.block_size - 1) // self.block_size

    def num_full_blocks(self) -> int:
        return self.num_tokens // self.block_size

    def block_token_ids(self, i: int) -> list[int]:
        s = i * self.block_size
        return self.token_ids[s : s + self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
