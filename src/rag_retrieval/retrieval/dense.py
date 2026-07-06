"""Recuperador denso por embeddings (variante propuesta 1).

Pendiente de implementación en el Hito 4. Usará ``sentence-transformers`` con un
modelo multilingüe ligero (p. ej. ``paraphrase-multilingual-MiniLM-L12-v2``) y
búsqueda por similitud coseno sobre embeddings precalculados del corpus.
"""

from __future__ import annotations

from rag_retrieval.retrieval.base import Retrieved, Retriever


class DenseRetriever(Retriever):
    """Recuperación densa por similitud coseno entre embeddings."""

    name = "dense"

    def index(self, corpus: list[dict]) -> None:  # noqa: D102 (ver Retriever)
        raise NotImplementedError("Se implementa en el Hito 4 (retriever denso).")

    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:  # noqa: D102
        raise NotImplementedError("Se implementa en el Hito 4 (retriever denso).")
