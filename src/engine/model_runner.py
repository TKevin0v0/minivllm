from __future__ import annotations

import torch
from torch import nn

from ..kv import ForwardContext, Sequence, set_forward_context
from ..pagedattention import slots_tensor
from ..sampler import Sampler
from ..kv.manager import KVManager


class ModelRunner:
    def __init__(
        self,
        model: nn.Module,
        kv: KVManager,
        sampler: Sampler,
        *,
        device: torch.device,
    ) -> None:
        self.model = model
        self.kv = kv
        self.sampler = sampler
        self.block_size = kv.block_size
        self.device = device

    def _tensor(self, data, dtype=torch.long) -> torch.Tensor:
        return torch.tensor(data, device=self.device, dtype=dtype)

    def _sample_one(self, logits: torch.Tensor, seq: Sequence) -> int:
        if seq.temperature is not None and seq.temperature <= 0:
            # t<=0开启贪心解码，直接取分数最大的token
            return int(torch.argmax(logits, dim=-1).item())

        def opt(v, dtype):
            return None if v is None else torch.tensor([v], device=self.device, dtype=dtype)

        token = self.sampler(
            logits,
            opt(seq.temperature, torch.float) if seq.temperature and seq.temperature > 0 else None,
            opt(seq.min_p, torch.float),
            opt(seq.top_p, torch.float),
            opt(seq.top_k, torch.int),
        )
        return int(token.item()) # 取出tensor的token_id,转为普通int返回

    @torch.inference_mode()
    def run_prefill_one(self, seq: Sequence) -> int:
        """ 单次 prefill 前向  """
        start = seq.num_cached_tokens # KV缓存token数
        end = start + seq.num_scheduled_tokens # 最后一个token下标
        tokens = seq.token_ids[start:end] # 待处理的token
        slots = slots_tensor(
            seq.block_table, start, end, self.block_size, device=self.device
        ) # 待处理token的物理位置
        set_forward_context(
            ForwardContext(
                is_prefill=True,
                store=self.kv.store,
                block_tables=[seq.block_table],
                slot_mapping=slots,
                query_lens=[end - start],
                context_lens=[end],
                cached_lens=[start],
            )
        )
        try:
            ids = self._tensor([tokens])
            positions = self._tensor([list(range(start, end))])
            h = self.model(ids, positions)
            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            return self._sample_one(logits, seq)
        finally:
            set_forward_context(None)

    @torch.inference_mode()
    def run_decode_batch(self, seqs: list[Sequence]) -> list[int]:
        """ 批量 decode 前向，一批多个生成中序列，每个 seq 只输入 1 个 token，并行推理，各自生成下 1 个 token """
        input_ids = [seq.last_token for seq in seqs] # 每个seq只输入一个token
        positions = [len(seq) - 1 for seq in seqs]
        slots = [
            seq.block_table[-1] * self.block_size + ((len(seq) - 1) % self.block_size)
            # 最后一个物理块位置加上偏移量，得到新算出的token对应的物理slot位置
            for seq in seqs
        ]
        set_forward_context(
            ForwardContext(
                is_prefill=False,
                store=self.kv.store,
                block_tables=[list(s.block_table) for s in seqs], # 每个seq的块表
                slot_mapping=self._tensor(slots),
                query_lens=[1] * len(seqs), # Q 长度
                context_lens=[len(s) for s in seqs], # 每条seq历史总token长度，根据这个读KV
            )
        )
        try:
            ids = self._tensor(input_ids).unsqueeze(-1)
            pos = self._tensor(positions).unsqueeze(-1)
            h = self.model(ids, pos)
            logits = self.model.compute_logits(h[:, -1:]).squeeze(1)
            # 取每个 batch 那条唯一的 token 的 hidden,做词表映射
            return [self._sample_one(logits[i : i + 1], seq) for i, seq in enumerate(seqs)]
            # 取出 batch 中第 i 个请求的 logits，结合该 seq 的采样参数采样出下一个 token id
            # 返回 list[int]，顺序和输入 seqs 对应
        finally:
            set_forward_context(None) # 清空全局 ForwardContext

    def run(self, prefills: list[Sequence], decodes: list[Sequence]) -> tuple[list[int], list[int]]:
        """ step()调用 优先执行 decode，保证正在生成中的 decode 请求持续推进，之后再执行 prefill 分片"""
        d_ids = self.run_decode_batch(decodes) if decodes else []
        p_ids = [self.run_prefill_one(s) for s in prefills] if prefills else []
        return p_ids, d_ids
