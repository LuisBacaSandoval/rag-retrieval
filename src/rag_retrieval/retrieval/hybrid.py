"""Recuperador híbrido con fusión Reciprocal Rank Fusion (variante propuesta 2).

Pendiente de implementación en el Hito 5. Fusiona los rankings de BM25 y del
recuperador denso mediante RRF, que combina posiciones (``rank``) en lugar de
puntajes crudos, precisamente porque los puntajes de ambos métodos no son
comparables entre sí.
"""

from __future__ import annotations

from rag_retrieval.retrieval.base import Retrieved, Retriever


class HybridRetriever(Retriever):
    """Fusión de recuperación léxica y densa mediante RRF."""

    name = "hybrid"

    def index(self, corpus: list[dict]) -> None:  # noqa: D102 (ver Retriever)
        raise NotImplementedError("Se implementa en el Hito 5 (híbrido con RRF).")

    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:  # noqa: D102
        raise NotImplementedError("Se implementa en el Hito 5 (híbrido con RRF).")
