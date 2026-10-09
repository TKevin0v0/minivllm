"""Bench vs official vLLM + MoE e2e latency.

Install official vLLM in a **separate** env (do not mix into this project's
Torch/FlashInfer stack)::

    # example
    uv venv ~/venvs/vllm-official
    uv pip install vllm --python ~/venvs/vllm-official/bin/python

Then::

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    # our engine
    PYTHONPATH=. python bench/moe_e2e.py --devices 1
    # compare (uses VLLM_BIN or python -c import vllm)
    PYTHONPATH=. python bench/vs_vllm.py --model dense --devices 1 \\
        --vllm-python ~/venvs/vllm-official/bin/python
    PYTHONPATH=. python bench/vs_vllm.py --model moe --devices 1 \\
        --vllm-python ~/venvs/vllm-official/bin/python

Metrics (both engines, same prompt / max_tokens / greedy)::

    ttft_ms, decode_tok_s, e2e_tok_s, tokens (for correctness spot-check)

MoE matrix (TP / EP / CUDA Graph)::

    bash bench/run_moe_matrix.sh
    # or single cell:
    PYTHONPATH=. python bench/moe_e2e.py --devices 0 --tp 1 --cg --moe-backend triton
    PYTHONPATH=. python bench/moe_e2e.py --devices 0,1 --tp 2 --ep --eager
"""
