# Long-request Stress Test Report

## Protocol

- Model: Qwen3-1.7B
- Code: NPU-adapted `vllm-v2-npu`
- Single prompt: `Explain how continuous batching improves LLM serving throughput and latency in detail.`
- Batch prompts: four fixed prompts covering continuous batching, KV cache paging, GPU/NPU scheduling, and benchmark methodology
- Generation: 256 new tokens per request
- TTFT: generation start to first yielded token
- TTOT: generation start to final yielded token
- Batch throughput: total generated tokens / batch wall-clock latency

## Stress results

| Metric | GPU RTX 3080 Ti | NPU Ascend 910 | GPU vs NPU |
|---|---:|---:|---:|
| Single TTFT | 253.784 ms | 219.506 ms | NPU 13.5% lower |
| Single TTOT | 9.826 s | 12.064 s | GPU 18.6% lower |
| Single decode | 26.743 tok/s | 21.613 tok/s | GPU 23.7% higher |
| 4-request batch tokens | 1024 | 1024 | Same workload |
| 4-request batch latency | 13.711 s | 18.900 s | GPU 27.5% lower |
| 4-request batch throughput | 74.682 tok/s | 54.180 tok/s | GPU 37.8% higher |

## Short baseline (32 tokens)

| Metric | GPU | NPU |
|---|---:|---:|
| TTFT | 248.919 ms | 224.943 ms |
| TTOT | 1.445 s | 1.648 s |
| Decode | 26.756 tok/s | 22.479 tok/s |
| 4-request batch throughput | ~63.4 tok/s | ~51.6 tok/s |
| 4-request batch latency | 2.02 s | 2.48 s |

## Interpretation

The long-request run is more representative of decode steady state than the 32-token smoke test. GPU has higher steady-state single-request and batch decode throughput, while NPU shows lower first-token latency in these runs. TTFT includes only generation timing after model initialization; model loading is excluded. Results are one run per configuration and should be repeated with warm-up and medians before publication as a production benchmark.

Raw logs: `gpu-new/stress.log`, `npu/stress-single.log`, `npu/stress-batch.log`; short baseline logs remain in `gpu-new/latency.log`, `gpu-new/test-batch.log`, `npu/unified-single.log`, and `npu/unified-batch.log`.
