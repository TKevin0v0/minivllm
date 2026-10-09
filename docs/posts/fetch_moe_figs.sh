#!/usr/bin/env bash
# 重新获取 docs/00~02 用到的论文 / 博客原图。
# 需要可访问 arxiv.org 和 huggingface.co。若本地配了代理，先
# export http_proxy / https_proxy。
#
# Switch Figure 3 的裁剪图 switch_fig3_routing.png 来自 PDF 第 6 页，
# 本脚本在有 pdftoppm / convert 时自动生成；没有就跳过。
#
# 用法（在 vllm-v5/ 下）：
#     bash docs/posts/fetch_moe_figs.sh

set -euo pipefail
cd "$(dirname "$0")"

fetch() {
  local out="$1" url="$2"
  echo "fetch $out"
  curl -sfL --max-time 60 -o "$out" "$url"
}

# Hugging Face《Mixture of Experts Explained》配图
# https://huggingface.co/blog/moe
HF="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe"
fetch hf_switch_layer.png     "$HF/00_switch_transformer.png"   # Switch 论文 Figure 2
fetch hf_shazeer_lstm.png     "$HF/01_moe_layer.png"            # Shazeer 稀疏门控层
fetch hf_moe_encoder.png      "$HF/02_moe_block.png"            # 稠密 → MoE → 设备放置
fetch hf_switch_layer2.png    "$HF/03_switch_layer.png"         # Switch 层另一版
fetch hf_ep_parallelism.png   "$HF/10_parallelism.png"          # Switch 论文 Figure 9
fetch hf_expert_matmuls.png   "$HF/11_expert_matmuls.png"       # batched / block-sparse GEMM

# Mixtral of Experts (Jiang et al., 2024) HTML 版插图
fetch mixtral_smoe.png \
  "https://arxiv.org/html/2401.04088v1/images/smoe.png"
fetch mixtral_routing_sample.png \
  "https://arxiv.org/html/2401.04088v1/images/routing-sample.png"
fetch mixtral_bench.png \
  "https://arxiv.org/html/2401.04088v1/images/231209_bench_combined.png"
fetch mixtral_scaling.png \
  "https://arxiv.org/html/2401.04088v1/images/231209_scaling.png"

fetch hf_experts_learning.png "$HF/05_experts_learning.png"

# DeepSeekMoE (Dai et al., 2024) HTML 版插图
fetch deepseek_x1.png "https://arxiv.org/html/2401.06066/x1.png"
fetch deepseek_x2.png "https://arxiv.org/html/2401.06066/x2.png"
fetch deepseek_x3.png "https://arxiv.org/html/2401.06066/x3.png"

# Switch Transformers Figure 1 / 3：从 PDF 裁出
# pdftoppm 输出名是 p3-03.png / p6-06.png（页码），不是 p3-1.png
if command -v pdftoppm >/dev/null && command -v convert >/dev/null; then
  tmp="$(mktemp -d)"
  echo "fetch switch.pdf (Figure 1 / 3 crop)"
  curl -sfL --max-time 90 -o "$tmp/switch.pdf" "https://arxiv.org/pdf/2101.03961.pdf"
  pdftoppm -png -r 150 -f 3 -l 3 "$tmp/switch.pdf" "$tmp/p3"
  # 第 3 页：完整 Figure 1（左缩放 + 右样本效率）+ 图注；起点不能太低，否则切掉曲线上半截
  convert "$tmp/p3-03.png" -crop 1210x515+30+580 +repage switch_fig1_scaling.png
  pdftoppm -png -r 160 -f 6 -l 6 "$tmp/switch.pdf" "$tmp/p6"
  # 完整 Figure 3（标题+示意图+图注）；勿切掉专家框顶部，也不要带上 §2.2
  convert "$tmp/p6-06.png" -crop 1307x720+28+555 +repage switch_fig3_routing.png
  rm -rf "$tmp"
else
  echo "skip Switch PDF crops (need pdftoppm + convert)"
fi

echo "done"
echo "本地示意图请再跑: python docs/posts/gen_moe_figs.py"
