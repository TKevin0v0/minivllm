# 512/1024 Token Performance and Precision Report

## Protocol
- Model: Qwen3-1.7B; identical NPU-adapted vllm-v2-npu code
- Single prompt: Explain how continuous batching improves LLM serving throughput and latency in detail.
- Batch: same four fixed prompts; greedy temperature=0.0
- Requested output lengths: 512 and 1024 new tokens
- TTFT: generation start to first yielded token; TTOT: generation start to final token

## Results
| Length / metric | GPU RTX 3080 Ti | NPU Ascend 910 |
|---|---:|---:|
| 512 TTFT | 384.636 ms | 205.616 ms |
| 512 TTOT | 26.466 s | 24.339 s |
| 512 single decode | 19.631 tok/s | 21.215 tok/s |
| 512 batch tokens | 2048 | 2048 |
| 512 batch latency | 37.179 s | 43.68 s |
| 512 batch throughput | 55.085 tok/s | 46.89 tok/s |
| 1024 TTFT | 44.228 ms* | 248.127 ms |
| 1024 TTOT | 52.246 s | 50.762 s |
| 1024 single decode | 19.616 tok/s | 20.272 tok/s |
| 1024 batch tokens | 4096 | 4096 |
| 1024 batch latency | 74.605 s | 95.53 s |
| 1024 batch throughput | 54.903 tok/s | 42.87 tok/s |

*GPU 1024 followed 512 in one process and benefited from warmed state; do not use as a cold-start TTFT claim.

## Precision check
Fixed prompt `What is the purpose of a KV cache in an LLM?`, 64 greedy tokens. GPU and NPU both produced 64 tokens with SHA-256 `7c56f77af144a6c23630304175e35fe43bea8cc05918d749e647ab381b7573f` and the same output prefix. This proves end-to-end greedy output byte equality for this probe, not per-layer/logit equality.

## Limitations
These are single-run engineering measurements. Repeat and report medians for publication. NPU 512 batch latency was captured interactively and is approximate. Raw GPU logs: `gpu-new/stress-512-1024-final.log`, `gpu-new/precision.log`. Raw NPU terminal outputs: `npu/stress-single.log`, `npu/stress-batch.log``.

