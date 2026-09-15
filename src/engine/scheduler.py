from __future__ import annotations
from collections import deque
from ..kv import KVManager, Sequence, SequenceStatus


class Scheduler:
    def __init__(
        self,
        kv: KVManager,
        *,
        max_num_seqs: int, # 最大并发序列数
        max_num_batched_tokens: int, # 单 batc h最大 token 数
        eos_token_id: int,
        mix_prefill_decode: bool = True,
    ) -> None:
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos_token_id
        self.mix_prefill_decode = mix_prefill_decode
        self.block_size = kv.block_size
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """ add_request() 调用 新请求加入等待队列 """
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        """ step() 调用，调用入口，返回 prefill 和 decode 列表"""
        if self.mix_prefill_decode:
            return self._schedule_mixed()
        return self._schedule_prefill_first()



    def _room(self, prefills: list[Sequence], decodes: list[Sequence]) -> bool:
        # 判断是否还有空余并发位置，不能超过最大并发数
        return len(prefills) + len(decodes) < self.max_num_seqs

    def _try_alloc(self, seq: Sequence) -> bool:
        # 尝试给 seq 分配 KV block
        if seq.block_table:
            return True
        if not self.kv.can_allocate(seq):
            return False
        self.kv.allocate(seq)
        return True

    def _promote_full_hit(self, seq: Sequence) -> None:
        # 前缀缓存命中完整 prompt 跳过 prefill 直接进入 running 准备 decode
        if not seq.block_table:
            if not self._try_alloc(seq):
                return
        seq.num_scheduled_tokens = 0
        seq.is_prefill = True # 临时保留 Ture 等到被 decode 调度选中再改写
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)

    def _schedule_decode(
        self,
        prefills: list[Sequence], # 本 batch 已经选出来的prefill 序列
        decodes: list[Sequence], # 同上 decodes 序列
        tokens_used: int, # 已经占用的 token 数
    ) -> int:
        """ 挑选 running 队列中的 decode 序列，加入本 batch ；KV 物理块 不足时需 preempt() """
        while self.running and self._room(prefills, decodes):
            if tokens_used + 1 > self.max_num_batched_tokens:
                break
            seq = self.running.popleft() # 弹出队列
            while not self.kv.can_append(seq):
                # 物理页不够分配，需要释放物理页
                if self.running:
                    # 还有别的 running 请求，驱逐队尾的 seq 释放它的 KV 页
                    self.preempt(self.running.pop())
                else:
                    # 没有别的running请求，只能驱逐当前seq
                    self.preempt(seq)
                    break
            else: # 不需要分配物理页(最后一个物理块没满)或者物理页足够分配
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.kv.may_append(seq) # 真正进行空闲页分配
                decodes.append(seq)
                tokens_used += 1 # 本轮输出 token 加1

        if decodes: # 把本batch要跑的decode，放回running队列头部，下一轮还能继续调度
            self.running.extendleft(reversed(decodes))
        return tokens_used

    def _pull_prefill_chunk(
        self,
        prefills: list[Sequence],
        decodes: list[Sequence],
        tokens_used: int,
    ) -> int:
        """ 从 waiting 取一个 prefill 分片，支持分块 prefill """
        if not self.waiting or not self._room(prefills, decodes):
            return tokens_used
        remaining = self.max_num_batched_tokens - tokens_used # 本batch剩余token预算
        if remaining <= 0:
            return tokens_used

        seq = self.waiting[0] # 取waiting队首元素 不弹出
        if not self._try_alloc(seq):
            # 尝试分配物理块，做prefix匹配，构建block_table
            return tokens_used

        need = seq.num_tokens - seq.num_cached_tokens # prefill还需要计算的token数
        if need <= 0: # 满命中，直接走decode
            self._promote_full_hit(seq)
            return tokens_used

        chunk = min(need, remaining)
        seq.num_scheduled_tokens = chunk
        seq.is_prefill = True
        prefills.append(seq)
        if seq.num_cached_tokens + chunk == seq.num_tokens:
            # 如果这一轮chunk跑完，prefill完成，走decode
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        # 这一轮chunk跑完，prefill没结束，继续留在waiting队首留给下轮处理
        # 下轮要处理的token数在postprocess处理
        return tokens_used + chunk

    def _schedule_prefill_first(self) -> tuple[list[Sequence], list[Sequence]]:
        """ prefill 优先调度：先把 prefill 批填满，没有 prefill 再跑 decode """
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        tokens_used = 0

        while self.waiting and self._room(prefills, decodes):
            remaining = self.max_num_batched_tokens - tokens_used # 剩余token额度
            if remaining <= 0:
                break
            seq = self.waiting[0]
            if not self._try_alloc(seq):
                break
            need = seq.num_tokens - seq.num_cached_tokens
            if need <= 0:
                self._promote_full_hit(seq)
                continue
            if remaining < need and prefills:
                break

            chunk = min(need, remaining)
            seq.num_scheduled_tokens = chunk
            seq.is_prefill = True
            tokens_used += chunk
            prefills.append(seq)
            if seq.num_cached_tokens + chunk == seq.num_tokens:
                # prefill跑完
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                break  # one partial chunk

        if prefills: # 如果本batch中，prefill没有跑完，直接返回，不允许一个batch同时跑prefill和decode
            return prefills, []
        # 本batch没有prefill要跑，走decode
        self._schedule_decode(prefills, decodes, 0)
        return [], decodes

    def _schedule_mixed(self) -> tuple[list[Sequence], list[Sequence]]:
        """ 混合调度模式：优先调度 decode ，再塞最多一个 prefill chunk """
        prefills: list[Sequence] = []
        decodes: list[Sequence] = []
        used = self._schedule_decode(prefills, decodes, 0)
        # 先把能跑的 decode 请求全部选进 batch，用完一部分 token 预算
        if decodes:
            # 本轮有decode，剩下的 batch token 额度，从 waiting 队列拿一个请求做 prefill 分片
            self._pull_prefill_chunk(prefills, decodes, used)
            return prefills, decodes
        # decode全部跑完了，进行prefill调度
        return self._schedule_prefill_first()

    def preempt(self, seq: Sequence) -> None:
        """抢占驱逐：把 running 中的 seq 踢回 waiting ；释放它的 KV 块，后续重新分配恢复"""
        if seq.status == SequenceStatus.FINISHED:
            return
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.kv.deallocate(seq, publish=False)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence], # 本轮执行的序列列表
        token_ids: list[int], # 模型输出token
        *,
        is_prefill: bool,
    ) -> list[Sequence]:
        """ step() 调用，模型推理完成之后的后处理：更新缓存计数、append token、判断结束、释放 KV """
        finished: list[Sequence] = []
        for seq, token_id in zip(seqs, token_ids):
            if seq.num_scheduled_tokens > 0:
                seq.num_cached_tokens += seq.num_scheduled_tokens # 本轮跑过的token累加到num_cached_tokens
                self.kv.sync_prefix(seq) # 更新 KV cache 满块登记到prefix
            seq.num_scheduled_tokens = 0 # 临时调度字段，每轮用完重置

            if is_prefill and seq.num_cached_tokens < seq.num_prompt_tokens:
                # prefill批次，并且prompt还没有跑完，说明分片了
                continue

            # prefill跑完或者decode
            seq.append_token(token_id) # 追加模型输出token，进入真正生成阶段
            hit_eos = (not seq.ignore_eos) and token_id == self.eos # 遇到结束token
            hit_len = seq.num_completion_tokens >= seq.max_tokens # 输出token满
            if hit_eos or hit_len:
                # 序列结束
                seq.status = SequenceStatus.FINISHED
                self.kv.deallocate(seq, publish=True) # 可复用式释放
                if seq in self.running: # 没跑完的也得强制跑完
                    self.running.remove(seq)
                finished.append(seq)
        return finished # 返回结束请求
