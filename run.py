"""vllm-v2 缁熶竴鍏ュ彛 鈥?Continuous Batching銆?

鐢ㄦ硶:
    python run.py                                    # 鍗曟潯鎺ㄧ悊锛堜娇鐢?config.yaml锛?
    python run.py --batch                            # 鎵归噺鎺ㄧ悊婕旂ず
    python run.py --model Qwen3-1.5B --prompt "浣犲ソ"  # CLI 瑕嗙洊閰嶇疆
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_utils import load_config, merge_cli  # noqa: E402

from src.config import EngineConfig  # noqa: E402
from src.engine import Engine  # noqa: E402
from src.model_hub import resolve_model_path  # noqa: E402
from src.sampling_params import SamplingParams  # noqa: E402

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "dtype": "auto",
    "device": "auto",
    "context_len": 2048,
    "block_size": 16,
    "prefix_backend": "hash",
    "max_num_seqs": 8,
    "max_num_batched_tokens": 4096,
    "mix_prefill_decode": True,
}


def run_single(cfg: dict) -> None:
    model_path = resolve_model_path(cfg["model"])
    print(f"Loading {model_path} (prefix={cfg['prefix_backend']}) ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            dtype=cfg["dtype"],
            device=cfg["device"],
            context_len=cfg["context_len"],
            block_size=cfg["block_size"],
            prefix_backend=cfg["prefix_backend"],
            max_num_seqs=cfg["max_num_seqs"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            mix_prefill_decode=cfg["mix_prefill_decode"],
        )
    )

    prompt = cfg["prompt"]
    print(f"Prompt: {prompt}\n{prompt}", end="", flush=True)
    start = time.perf_counter()
    n = 0
    for tok in engine.generate(
        prompt,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
    ):
        print(tok, end="", flush=True)
        n += 1
    elapsed = time.perf_counter() - start
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")


def run_batch(cfg: dict) -> None:
    model_path = resolve_model_path(cfg["model"])
    print(f"Loading {model_path} for continuous batching ...")
    engine = Engine(
        EngineConfig(
            model=str(model_path),
            prefix_backend=cfg["prefix_backend"],
            device=cfg["device"],
            max_num_seqs=cfg["max_num_seqs"],
            max_num_batched_tokens=cfg["max_num_batched_tokens"],
            mix_prefill_decode=cfg["mix_prefill_decode"],
        )
    )

    prompts = [
        "Hello, my name is",
        "Once upon a time",
        "The capital of France is",
        "In machine learning,",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg["max_new_tokens"])
    t0 = time.perf_counter()
    outs = engine.generate_batch(prompts, sp)
    dt = time.perf_counter() - t0
    for i, (prompt, text) in enumerate(zip(prompts, outs)):
        print(f"\n[{i}] {prompt!r}\n    鈫?{text!r}")
    total_tok = sum(len(engine.tokenizer.encode(o)) for o in outs)
    print(f"\n{len(prompts)} requests, ~{total_tok} toks in {dt:.2f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="vllm-v2 Continuous Batching")
    p.add_argument("--config", default=None, help="YAML 閰嶇疆鏂囦欢璺緞")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--prefix-backend", choices=("none", "hash", "radix"), default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--batch", action="store_true", help="杩愯鎵归噺鎺ㄧ悊婕旂ず")
    args = p.parse_args()

    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path, DEFAULTS)
    merge_cli(
        cfg,
        args,
        ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k", "block_size", "prefix_backend", "max_num_seqs", "device"),
    )

    if args.batch:
        run_batch(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()
