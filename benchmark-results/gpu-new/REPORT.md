# GPU Test Report: NPU-adapted v2 code

## Environment

- Host: `connect.nmb2.seetacloud.com:40423`
- GPU: NVIDIA GeForce RTX 3080 Ti, 12 GB
- Driver: 595.71.05
- CUDA reported by `nvidia-smi`: 13.2
- Python: `/root/miniconda3/bin/python3.12` (3.12.3)
- PyTorch: remote native environment; CUDA device available (see `torch.log`)
- Model: `/root/models/models/Qwen--Qwen3-1.7B/snapshots/master`
- Code: `/root/bench/vllm-v2-npu` (uploaded copy of the NPU-adapted v2 code)

## Commands

Single request:

```bash
cd /root/bench/vllm-v2-npu
PYTHONPATH=/root/bench/vllm-v2-npu /root/miniconda3/bin/python3.12 run.py --model /root/models/models/Qwen--Qwen3-1.7B/snapshots/master --device cuda --max-new-tokens 32 --prompt 'Hello, my name is'
```

Batch:

```bash
cd /root/bench/vllm-v2-npu
PYTHONPATH=/root/bench/vllm-v2-npu /root/miniconda3/bin/python3.12 run.py --model /root/models/models/Qwen--Qwen3-1.7B/snapshots/master --device cuda --max-new-tokens 32 --batch
```

## Measured results

| Test | Result |
|---|---:|
| Single request decode | 20.0 tok/s (32 tokens in 1.60 s) |
| 4-request batch aggregate | ~63.4 tok/s (about 128 tokens in 2.02 s) |
| 4-request batch total latency | 2.02 s |
| TTFT | Not exposed by this `run.py`; no value inferred |

Raw logs: `test-native.log`, `test-batch.log`, `env.log`, `torch.log`, `model.log`, and `install-mirror.log`.
