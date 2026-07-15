"""Fixed-size token chunker with overlap.

Port of `06_ChunkingStrategies/FixedSizeChunker.cs`.
"""

from __future__ import annotations

import tiktoken


class FixedSizeChunker:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._tokenizer = tiktoken.get_encoding(encoding_name)

    def create_chunks(
        self,
        text: str,
        chunk_size: int = 512,
        overlap_percentage: float = 25.0,
    ) -> list[str]:
        overlap_size = int(chunk_size * overlap_percentage / 100.0)
        stride = chunk_size - overlap_size
        if stride <= 0:
            raise ValueError(
                f"overlap_percentage ({overlap_percentage}%) produces a non-positive "
                "stride. Keep it below 100%."
            )

        token_ids = self._tokenizer.encode(text)
        chunks: list[str] = []
        for start in range(0, len(token_ids), stride):
            end = min(start + chunk_size, len(token_ids))
            decoded = self._tokenizer.decode(token_ids[start:end])
            if decoded:
                chunks.append(decoded)
            if end == len(token_ids):
                break
        return chunks

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))
