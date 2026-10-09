# v5 MoE 文档

v4 写清了词表并行和线性层并行。v5 多出来的是 Mixture of Experts：把 Transformer 里的稠密 MLP 换成一组专家，每个 token 只走其中几个。

阅读顺序：先看 MoE 在算什么，再看一层由哪些模块组成、基础核怎么写、融合路径怎么加速，然后是多卡切法，最后落到本仓库文件。

| 文档 | 写什么 |
| --- | --- |
| [00-MoE概述.md](00-MoE概述.md) | 从 Jacobs 到 Qwen3-MoE：稀疏门控、接到 FFN、Switch、Mixtral、DeepSeek、Qwen1.5/2/2.5 再到 Qwen3；常考题 |
| [01-路由与融合计算.md](01-路由与融合计算.md) | 从 naive 双重循环到 align / grouped GEMM，再串回 v5 前向 |
| [02-专家并行.md](02-专家并行.md) | MoE TP 与 EP、v5 不搬 token、共享专家与 AR 顺序、常考题 |
| [03-v5实现.md](03-v5实现.md) | `src/moe/` 文件地图、权重加载、后端、与官方差异、常考题 |

代码入口：[`src/moe/`](../src/moe/)。模型：[`src/compute/models/qwen3_moe.py`](../src/compute/models/qwen3_moe.py)。测试默认模型见 [`tests/README.md`](../tests/README.md)：Qwen1.5-MoE-A2.7B 跑 TP/EP，Qwen3-30B-A3B 做大模型冒烟。

VL、OpenAI serving、DeepEP all-to-all 不在这套文档里。稠密 Qwen3 的并行和词表仍走 v4。

配图：`posts/` 里 `hf_` / `mixtral_` / `deepseek_` / `switch_` 前缀为论文或博客原图，可用 `bash docs/posts/fetch_moe_figs.sh` 重拉。`dense_vs_sparse.png` 一类为本实现示意图，用 `python docs/posts/gen_moe_figs.py` 生成。
