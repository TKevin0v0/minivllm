# Benchmark Results

The final unified table is `UNIFIED_COMPARISON.md`. All raw data is in this folder:

- GPU: `gpu-new/latency.log`, `gpu-new/test-batch.log`, `gpu-new/env.log`, `gpu-new/torch.log`
- NPU: `npu/unified-single.log`, `npu/unified-batch.log`, `npu/environment.md`, `npu/env-current.log`

Measured under the same Qwen3-1.7B prompt and max_new_tokens=32 protocol:

- GPU: TTFT 248.919 ms, TTOT 1444.903 ms, decode 26.756 tok/s, batch 63.4 tok/s, batch latency 2.02 s.
- NPU: TTFT 224.943 ms, TTOT 1648.478 ms, decode 22.479 tok/s, batch 51.6 tok/s, batch latency 2.48 s.
