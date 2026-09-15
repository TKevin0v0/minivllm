# vllm-v2 — Continuous Batching

> 在 v1（KV Cache + 前缀）之上加入多请求调度，引入"控制面 / 数据面 / 计算面"三面分离架构。

---

## 代码架构

```
vllm-v2/
├── run.py                        # 改写：+ --batch
├── config.yaml                   # 改写：+ max_num_seqs / max_num_batched_tokens / mix_prefill_decode
├── src/
│   ├── config.py                 # 改写：+ max_num_seqs / max_num_batched_tokens
│   ├── sampling_params.py        # 新增：每请求采样参数 dataclass
│   ├── engine/                   # 新增：Engine 拆包（v1 是单文件 src/engine.py）
│   │   ├── scheduler.py          #   新增：waiting/running、chunked prefill、PD 混跑
│   │   ├── model_runner.py       #   新增：组 batch、绑 context、跑模型、采样
│   │   └── engine.py             #   改写：step = schedule → run → postprocess，generate_batch
│   ├── kv/
│   │   ├── sequence.py           # 改写：多请求 Sequence 生命周期
│   │   ├── manager.py            # 改写：多请求 admission（can_allocate / can_append / may_append）
│   │   └── gather.py             # 新增：_write_kv / _gather_kv
│   ├── pagedattention/
│   │   └── store.py              # 改写：write 支持 decode batch 展平
│   └── layers/
│       └── attention.py          # 改写：prefill chunk + decode 合批
└── docs/                         # 设计文档
```

---

## 相对 v1 新增

| 新增能力 | 对应模块 | 说明 |
|---------|---------|------|
| Continuous Batching | `src/engine/scheduler.py` | waiting/running 队列，每步 schedule → run → postprocess |
| Chunked Prefill | 同上 | 长 prompt 切块，与 decode 混合调度 |
| PD 混跑 | 同上 | `mix_prefill_decode`：decode 优先 + 剩余 budget 给 prefill |
| SamplingParams | `src/sampling_params.py` | 独立采样参数类（temperature / max_tokens / top_p） |
| `generate_batch()` | `src/engine/engine.py` | 批量请求并行生成 |
| 三面分离 | `scheduler.py` / `model_runner.py` / `layers/` | 控制面 / 数据面 / 计算面解耦 |
| 序列管理 | `src/kv/sequence.py` | 多请求 Sequence 生命周期 |

---

## 快速开始

```bash
cd vllm-v2

# 单条推理
python run.py --model Qwen3-0.6B

# 批量推理
python run.py --batch

# 性能对比（v0/v1/v2）
python -m bench.compare --model Qwen3-0.6B

# Chunked P+D 混跑公平性测试
python -m bench.compare_chunked --model Qwen3-0.6B --budget 32 --long-prompt 512 --conc 2,4
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [00-V2版本设计解释.md](docs/00-V2版本设计解释.md) | Continuous Batching 动机 |
| [01-相比v1的增改.md](docs/01-相比v1的增改.md) | v1 → v2 逐个差异文件清单 |
| [02-代码架构与三面分离.md](docs/02-代码架构与三面分离.md) | 控制面 / 数据面 / 计算面架构详解 |
