# Mini vLLM (NPU-adapted v2)

This project is a local copy of the NPU-adapted `vllm-v2-npu` implementation, renamed for project and resume use as `minivllm`.

## Layout

- `src/`, `utils/`, `run.py`: adapted implementation
- `benchmark-results/`: all GPU/NPU benchmark reports and raw logs
- `benchmark-results/STRESS_COMPARISON.md`: long-request pressure comparison
- `benchmark-results/UNIFIED_COMPARISON.md`: unified short-request comparison

The original `vllm-v2-npu` source directory remains unchanged.
