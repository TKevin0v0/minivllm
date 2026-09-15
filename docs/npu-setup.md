# Ascend NPU setup

The v2 runtime supports `device: auto|npu|cuda|cpu`. On an Ascend host, use
the virtual environment supplied with the image and ensure `torch_npu` matches
both PyTorch and CANN before starting the engine.

```bash
cd /mnt/host-model/ncx/vllm-v2
source /mnt/host-model/ncx/venv-sglang-npu/bin/activate
python -c 'import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available())'
python run.py --device npu --model /mnt/model/<model> --max-new-tokens 8
```

The first validation target is single-card eager inference. Continuous
batching and larger models should be enabled only after the basic model,
SDPA, and KV-cache checks pass.
