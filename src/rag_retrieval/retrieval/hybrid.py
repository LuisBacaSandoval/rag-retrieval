"""Recuperador híbrido con Reciprocal Rank Fusion (variante propuesta 2).

Fusiona BM25 y denso por posición, no por puntaje (no comparables); sección 2.4.
"""

from __future__ import annotations

from rag_retrieval.retrieval.base import Retrieved, Retriever
from rag_retrieval.retrieval.bm25 import BM25Retriever
from rag_retrieval.retrieval.dense import DenseRetriever


def rrf_fusion(ranked_lists: list[list[str]], c: int = 60) -> list[tuple[str, float]]:
    """Cada doc suma ``1/(c + posición)`` de cada lista; empate por primera aparición."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for position, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + position)

    return sorted(scores.items(), key=lambda item: -item[1])


class HybridRetriever(Retriever):
    """Fusión de recuperación léxica (BM25) y densa mediante RRF."""

    name = "hybrid"

    def __init__(
        self,
        c: int = 60,
        pool: int = 50,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
    ) -> None:
        self.c = c
        self.pool = pool  # candidatos por método antes de fusionar
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()

    def index(self, corpus: list[dict]) -> None:  # noqa: D102 (ver Retriever)
        self.bm25.index(corpus)
        self.dense.index(corpus)

    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:  # noqa: D102
        if k <= 0:
            raise ValueError("k debe ser mayor que cero")

        pool = max(self.pool, k)
        lexico = [r.doc_id for r in self.bm25.retrieve(query, k=pool)]
        denso = [r.doc_id for r in self.dense.retrieve(query, k=pool)]

        fusionados = rrf_fusion([lexico, denso], c=self.c)[:k]
        return [
            Retrieved(doc_id=doc_id, score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(fusionados, start=1)
        ]
