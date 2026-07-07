"""Recuperadores del proyecto: interfaz común y variantes (BM25, denso, híbrido)."""

from rag_retrieval.retrieval.base import Retrieved, Retriever
from rag_retrieval.retrieval.bm25 import BM25Retriever

__all__ = ["Retrieved", "Retriever", "BM25Retriever"]
