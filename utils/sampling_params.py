from __future__ import annotations
from dataclasses import dataclass

__all__ = ["SamplingParams"]

@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 128
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 0 or self.temperature < 0:
            raise ValueError("max_tokens and temperature must be non-negative")
