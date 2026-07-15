"""Shared helpers for semantic / hierarchical chunkers.

Port of `06_ChunkingStrategies/SemanticChunkerBase.cs`.
"""

from __future__ import annotations

import math
import re

import tiktoken


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SemanticChunkerBase:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    @staticmethod
    def split_into_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Vectors must have the same dimension")
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        return 0.0 if denom == 0.0 else dot / denom
