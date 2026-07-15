"""In-memory vector store with cosine-similarity search.

Direct port of the reference repo's `Shared/InMemoryVectorDb.cs`,
`Shared/VectorSearchRecord.cs`, and `Shared/VectorSearchResult.cs`
(https://github.com/deployed-in-azure/RAG/tree/main/RAG/Shared).
"""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VectorSearchRecord:
    id: str
    vector: list[float]
    data: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    similarity: float
    data: dict[str, str]


class InMemoryVectorDb:
    def __init__(self, supported_vector_dimension: int = 1536) -> None:
        self._supported_vector_dimension = supported_vector_dimension
        self._vectors: dict[str, VectorSearchRecord] = {}

    def index(self, vector_document: VectorSearchRecord) -> None:
        if len(vector_document.vector) != self._supported_vector_dimension:
            raise ValueError(
                f"Invalid vector dimension. The only supported dimension is "
                f"{self._supported_vector_dimension}."
            )

        if vector_document.id in self._vectors:
            raise ValueError(f"A document with ID '{vector_document.id}' already exists.")

        self._vectors[vector_document.id] = vector_document

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        if len(query_vector) != self._supported_vector_dimension:
            raise ValueError(
                f"Invalid vector dimension. The only supported dimension is "
                f"{self._supported_vector_dimension}."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        scored = [
            (record, self._cosine_similarity(query_vector, record.vector))
            for record in self._vectors.values()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return [
            VectorSearchResult(id=record.id, similarity=similarity, data=record.data)
            for record, similarity in scored[:top_k]
        ]

    @staticmethod
    def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        dot_product = 0.0
        sum_squares_a = 0.0
        sum_squares_b = 0.0

        for a, b in zip(vector_a, vector_b):
            dot_product += a * b
            sum_squares_a += a * a
            sum_squares_b += b * b

        vector_a_length = math.sqrt(sum_squares_a)
        vector_b_length = math.sqrt(sum_squares_b)
        denominator = vector_a_length * vector_b_length

        cosine = 0.0 if denominator == 0.0 else dot_product / denominator
        return round(cosine, 2)
