from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import torch
from transformers import AutoTokenizer

from ..config import EngineConfig
from ..device import empty_cache, resolve_device
from ..kv import KVManager, Sequence
from ..model import Qwen3ForCausalLM
from ..model_loader import load_model
from ..sampling_params import SamplingParams
from ..sampler import Sampler
from .model_runner import ModelRunner
from .scheduler import Scheduler


class Engine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        print(f"Using device: {self.device}")
        self.dtype = self._resolve_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.context_len = config.context_len
        self.block_size = config.block_size

        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.model = self.model.to(dtype=self.dtype, device=self.device).eval()

        self.sampler = Sampler()
        self._init_kv()
        self.last_num_cached_tokens = 0

    def _init_kv(self) -> None:
        """ 鍒濆鍖朘V缂撳瓨锛岃姹傝皟搴︿互鍙婃ā鍨嬫墽琛?"""
        # 绠＄悊鐗╃悊block鍐呭瓨锛屽墠缂€缂撳瓨锛屽澶栫粰Scheduler鎻愪緵block鍒嗛厤鎺ュ彛锛岀粰ModelRunner鎻愪緵KV缂撳瓨寮犻噺
        self.kv = KVManager(
            self.model,
            num_blocks=self.config.num_kv_blocks,
            block_size=self.config.block_size,
            prefix_backend=self.config.prefix_backend,
            device=self.device,
            dtype=self.dtype,
        )
        # 璇锋眰璋冨害鍣紝璇锋眰鎺掗槦锛岃祫婧愬垽鏂紝鐘舵€佹洿鏂?
        self.scheduler = Scheduler(
            self.kv,
            max_num_seqs=self.config.max_num_seqs, # 鏈€澶ц姹傛暟
            max_num_batched_tokens=self.config.max_num_batched_tokens, # 鍗曡疆token涓婇檺
            eos_token_id=self.eos_token_id, # 缁撴潫绗︼紝缁撳悎is_finished
            mix_prefill_decode=self.config.mix_prefill_decode, # 璋冨害绛栫暐
        )
        # 妯″瀷鎵ц鍣?鎺ユ敹Scheduler杈撳嚭鐨勫簭鍒楀垪琛ㄥ悜鍓嶄紶鎾苟杩斿洖鏂扮殑token_id
        self.model_runner = ModelRunner(
            self.model, self.kv, self.sampler, device=self.device
        )

    def reset_kv(
        self, prefix_backend: Literal["none", "hash", "radix"] | None = None
    ) -> None:
        """ 杩愯鏃堕噸缃暣濂?KV 缂撳瓨瀛愮郴缁?"""
        if prefix_backend is not None:
            if prefix_backend not in ("none", "hash", "radix"):
                raise ValueError(f"invalid prefix_backend: {prefix_backend!r}")
            self.config.prefix_backend = prefix_backend # 鐑垏鎹㈠墠缂€缂撳瓨绛栫暐
        self._init_kv() # 鍒濆鍖?KVManager銆丼cheduler銆丮odelRunner
        self.last_num_cached_tokens = 0 # 鍓嶇紑缂撳瓨鍛戒腑鐨則oken鏁伴噺
        empty_cache(self.device)

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.dtype != "auto":
            return getattr(torch, self.config.dtype)
        cfg = self.config.hf_config
        dt = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
        return dt if isinstance(dt, torch.dtype) else torch.float16

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        """ 灏?prompt 鍖呰鎴?Sequence 瀵硅薄锛屽苟杩涘叆璋冨害鍣ㄦ帓闃?"""
        token_ids = (
            list(prompt) if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        )
        if len(token_ids) > self.context_len:
            token_ids = token_ids[-self.context_len :]
        seq = Sequence.from_prompt(
            token_ids, block_size=self.block_size, sampling=sampling_params
        )
        self.scheduler.add(seq)
        return seq

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """ """
        prefills, decodes = self.scheduler.schedule()
        # prefills 闇€瑕佹墽琛宲refill鐨勫簭鍒?
        # decodes 姝ｅ湪鍋氳凯浠ｇ敓鎴愮殑搴忓垪

        if not prefills and not decodes:
            return [], 0

        # 鏉ヨ嚜 allocate() 鐨刾refill闃舵token鍛戒腑鐨勬暟閲?
        self.last_num_cached_tokens = self.kv.last_num_cached_tokens

        prefill_toks = sum(s.num_scheduled_tokens for s in prefills) # 鏈疆鎵€鏈塸refill璺戠殑token鏁伴噺
        metric = prefill_toks + len(decodes) # 鏈疆鎵€鏈夌殑token鏁伴噺

        # 浼樺厛鎵ц decode锛屼繚璇佹鍦ㄦ祦寮忚緭鍑虹殑璇锋眰鎸佺画鍚戝墠鎺ㄨ繘锛涗箣鍚庡啀鎵ц prefill 鍒嗙墖
        p_ids, d_ids = self.model_runner.run(prefills, decodes)

        finished: list = []
        # 妯″瀷璺戝畬鍚庡鐞? 杩藉姞鐢熸垚鐨則oken_id, 鍒嗛厤鏂扮敓鎴恡oken鐨勭墿鐞嗗湴鍧€
        # 杩斿洖鐪熸鎵ц缁撴潫鐨剆eq鍒楄〃
        if decodes:
            finished.extend(
                self.scheduler.postprocess(decodes, d_ids, is_prefill=False)
            )
        if prefills:
            finished.extend(
                self.scheduler.postprocess(prefills, p_ids, is_prefill=True)
            )
        # 姝ｅ父鎯呭喌涓嬶紝prefill鐨剆eq鏄笉浼氱粨鏉熺殑锛屽洜涓鸿繕瑕佺户缁璬ecode鐢熸垚
        # 浣嗘槸濡傛灉prompt杩囬暱瓒呰繃闄愬埗token闀垮害鎴栬€呴亣鍒癳os,浼氭彁鍓嶇粨鏉?

        outputs = [
            (seq.seq_id, seq.token_ids[seq.num_prompt_tokens :]) for seq in finished
        ] # 杩斿洖鍗曡姹傜殑璇锋眰id锛宒ecode杈撳嚭token
        return outputs, metric


    def generate_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """ 澶氳姹傚澶朅PI """
        # 澶勭悊閲囨牱鍙傛暟锛屾瘡鏉rompt瀵瑰簲涓€濂楅噰鏍烽厤缃?
        if sampling_params is None:
            # 閲囩敤榛樿鍙傛暟
            sps = [SamplingParams() for _ in prompts]
        elif isinstance(sampling_params, (list, tuple)):
            # 澶歱rompt锛岄噰鏍烽厤缃暟閲忓繀椤诲拰prompt鏁伴噺瀵归綈
            sps = list(sampling_params)
            if len(sps) != len(prompts):
                raise ValueError("sampling_params length must match prompts")
        else:
            # 鍗昿rompt
            sps = [sampling_params for _ in prompts]

        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, sps)]
        # seqs 淇濆瓨 prompt 杈撳叆椤哄簭
        finished_text: dict[int, str] = {}
        while not self.scheduler.is_finished():
            outputs, _ = self.step()
            # outputs鐨剆eq椤哄簭鏄粨鏉熼『搴?
            for seq_id, completion_ids in outputs:
                finished_text[seq_id] = self.tokenizer.decode(completion_ids)
        return [finished_text[seq.seq_id] for seq in seqs]
        # 淇濊瘉杈撳嚭椤哄簭鍜岃緭鍏ヤ竴鑷?

    def generate(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> Iterator[str]:
        """ 鍗曡姹傚澶朅PI """
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_new_tokens,
        )
        seq = self.add_request(prompt, sp)
        emitted = 0 # 杈撳嚭 token 璁℃暟鍣?
        while not seq.is_finished:
            self.step() # 妯″瀷璋冨害 + GPU 鎺ㄧ悊 + postprocess
            while emitted < seq.num_completion_tokens:
                # 鏂皌oken鐢熸垚锛屼絾鏄繕娌℃湁杈撳嚭
                tid = seq.token_ids[seq.num_prompt_tokens + emitted]
                emitted += 1
                yield self.tokenizer.decode([tid]) # 杈撳嚭
