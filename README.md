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

