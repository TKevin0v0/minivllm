# GPU Benchmark Failure

The GPU host is reachable and CUDA is functional, but no model weights are present. Searches under `/root`, `/autodl-fs`, `/autodl-pub`, `/data`, and `/workspace` found no Qwen `config.json` or `.safetensors` files; `/root/autodl-fs` and `/root/autodl-pub` are empty.

Verified environment: NVIDIA GeForce RTX 3080 Ti (12 GB), driver 595.71.05, CUDA 13.2, Python 3.12.3, PyTorch 2.12.1+cu130, CUDA available `True`, one device.

TTFT, single-request decode tokens/s, four-request batch throughput, and total latency were not run. No GPU performance value is inferred or fabricated.

Next step: place the same Qwen3-1.7B weights used by the NPU baseline in an explicit directory on the GPU host, then rerun with identical prompt, generation length, and batch settings.
