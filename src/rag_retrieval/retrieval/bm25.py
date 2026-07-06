"""Recuperador léxico BM25 (línea base).

Pendiente de implementación en el Hito 3. Se apoyará en ``rank-bm25`` (y
opcionalmente TF-IDF de scikit-learn como segunda referencia léxica).
"""

from __future__ import annotations

from rag_retrieval.retrieval.base import Retrieved, Retriever


class BM25Retriever(Retriever):
    """Línea base léxica basada en Okapi BM25."""

    name = "bm25"

    def index(self, corpus: list[dict]) -> None:  # noqa: D102 (ver Retriever)
        raise NotImplementedError("Se implementa en el Hito 3 (línea base BM25).")

    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:  # noqa: D102
        raise NotImplementedError("Se implementa en el Hito 3 (línea base BM25).")
