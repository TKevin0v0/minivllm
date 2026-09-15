from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Literal

from transformers import AutoConfig, PretrainedConfig


@dataclass
class EngineConfig:
    model: str
    context_len: int = 2048
    dtype: str = "auto"
    device: str = "auto"

    # KV / prefix (v1 新增)
    prefix_backend: Literal["none", "hash", "radix"] = "hash"
    block_size: int = 16
    num_kv_blocks: int = 0  # 0 → derive

    # continuous batching(v2 新增)
    max_num_seqs: int = 8  # 最大请求数量
    max_num_batched_tokens: int = 4096 # 单条batch最大token数
    # True: decode + prefill chunk 同一步；False: 旧 Prefill-first
    mix_prefill_decode: bool = True
    hf_config: PretrainedConfig = field(init=False)

    def __post_init__(self) -> None:
        self.model = os.path.expanduser(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        if self.device not in ("auto", "npu", "cuda", "cpu"):
            raise ValueError(f"device must be auto|npu|cuda|cpu, got {self.device!r}")
        if self.prefix_backend not in ("none", "hash", "radix"):
            raise ValueError(
                f"prefix_backend must be none|hash|radix, got {self.prefix_backend!r}"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be > 0")
        if self.num_kv_blocks <= 0:
            pages = math.ceil(self.context_len / self.block_size)
            self.num_kv_blocks = max(pages * self.max_num_seqs, pages + 8)
