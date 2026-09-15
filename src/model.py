from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from .layers.activation import SiluAndMul
from .layers.attention import Attention
from .layers.layernorm import RMSNorm
from .layers.rotary import RotaryEmbedding
from .model_loader import default_weight_loader

_QKV_SHARDS = (
    ("q_proj", "qkv_proj", lambda q, kv: 0, lambda q, kv: q),
    ("k_proj", "qkv_proj", lambda q, kv: q, lambda q, kv: kv),
    ("v_proj", "qkv_proj", lambda q, kv: q + kv, lambda q, kv: kv),
)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1))
        )


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * self.head_dim,
            bias=qkv_bias,
        )
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, self.head_dim, max_position, rope_theta
        )
        self.attn = Attention(
            num_heads, self.head_dim, self.head_dim**-0.5, num_kv_heads, layer_id
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden_states).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = self.q_norm(q.unflatten(-1, (-1, self.head_dim))).flatten(-2)
        k = self.k_norm(k.unflatten(-1, (-1, self.head_dim))).flatten(-2)
        q, k = self.rotary_emb(positions, q, k)

        b, s, _ = q.shape
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_kv_heads, self.head_dim)
        v = v.view(b, s, self.num_kv_heads, self.head_dim)
        return self.o_proj(self.attn(q, k, v).flatten(-2))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_id: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            max_position=config.max_position_embeddings,
            rope_theta=float(getattr(config, "rope_theta", 1_000_000)),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=bool(getattr(config, "attention_bias", False)),
            head_dim=getattr(config, "head_dim", None),
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        h, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, residual = layer(positions, h, residual)
        return self.norm(h, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        attn0 = self.model.layers[0].self_attn
        q_size, kv_size = attn0.q_size, attn0.kv_size

        for name, w in weights:
            merged = False
            for hf, our, off_fn, size_fn in _QKV_SHARDS:
                if hf not in name or "qkv_proj" in name:
                    continue
                fused = name.replace(hf, our)
                param = params[fused]
                off, size = off_fn(q_size, kv_size), size_fn(q_size, kv_size)
                param.data[off : off + size].copy_(w)
                loaded.add(fused)
                merged = True
                break
            if merged:
                continue
            if name in params:
                default_weight_loader(params[name], w)
                loaded.add(name)

        if "lm_head.weight" not in loaded and "model.embed_tokens.weight" in loaded:
            default_weight_loader(
                params["lm_head.weight"], params["model.embed_tokens.weight"]
            )
            loaded.add("lm_head.weight")

        missing = set(params) - loaded
        if missing:
            print(f"[WARN] {len(missing)} params not loaded: {sorted(missing)[:8]}...")
