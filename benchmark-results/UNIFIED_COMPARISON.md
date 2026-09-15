# Unified GPU/NPU Benchmark Comparison

Protocol: Qwen3-1.7B, prompt `Hello, my name is`, max new tokens 32, NPU-adapted `vllm-v2-npu` code, single request plus 4-request batch.

| Metric | GPU (RTX 3080 Ti) | NPU (Ascend 910) |
|---|---:|---:|
| Single TTFT | 248.919 ms | 224.943 ms |
| Single TTOT | 1444.903 ms | 1648.478 ms |
| Single decode | 26.756 tok/s | 22.479 tok/s |
| 4-request batch throughput | ~63.4 tok/s | ~51.6 tok/s |
| 4-request batch total latency | 2.02 s | 2.48 s |

## Commands

Both sides ran the NPU-adapted v2 code with Qwen3-1.7B, the same prompt, max_new_tokens=32, and the same four batch prompts. TTFT is generation start to first yielded token; TTOT is generation start to final token.

Raw logs: GPU `gpu-new/latency.log`, `gpu-new/test-batch.log`; NPU `npu/unified-single.log`, `npu/unified-batch.log`.

Batch throughput is approximately 128 generated tokens divided by total batch latency. These are one-run engineering measurements; collect repeated runs and medians before making production-level claims.
