"""Recuperador léxico Okapi BM25 (línea base).

Usa ``rank_bm25`` según el patrón del Cuaderno 21 (semana 10); teoría en el
cuaderno técnico, sección 2.1.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi

from rag_retrieval.retrieval.base import Retrieved, Retriever

_PALABRA = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Minúsculas, sin tildes y palabras ``\\w+``; se aplica igual a corpus y consultas."""
    descompuesto = unicodedata.normalize("NFKD", text.lower())
    sin_tildes = "".join(ch for ch in descompuesto if not unicodedata.combining(ch))
    return _PALABRA.findall(sin_tildes)


class BM25Retriever(Retriever):
    """Línea base léxica: ``k1`` satura la frecuencia de término y ``b`` normaliza por longitud."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    def index(self, corpus: list[dict]) -> None:  # noqa: D102 (ver Retriever)
        if not corpus:
            raise ValueError("El corpus no puede estar vacío")

        self._doc_ids = [doc["doc_id"] for doc in corpus]
        tokenized = [tokenize(doc["text"]) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:  # noqa: D102
        if self._bm25 is None:
            raise RuntimeError("Hay que llamar a index() antes de retrieve()")

        if k <= 0:
            raise ValueError("k debe ser mayor que cero")

        tokens = tokenize(query)
        scores = np.asarray(self._bm25.get_scores(tokens))

        # Orden estable: ante empate de puntaje gana el documento que aparece antes.
        limit = min(k, len(self._doc_ids))
        orden = np.argsort(-scores, kind="stable")[:limit]

        return [
            Retrieved(doc_id=self._doc_ids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(orden, start=1)
        ]
