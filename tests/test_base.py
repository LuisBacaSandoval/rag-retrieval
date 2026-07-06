"""Pruebas de contrato de la interfaz común de recuperadores.

Verifican que el esqueleto del proyecto es coherente antes de implementar los
métodos: la interfaz importa, las tres variantes la heredan y comparten la firma
esperada, y el dato ``Retrieved`` tiene los campos que la comparación necesita.
"""

from __future__ import annotations

import inspect

import pytest

from rag_retrieval.retrieval.base import Retrieved, Retriever
from rag_retrieval.retrieval.bm25 import BM25Retriever
from rag_retrieval.retrieval.dense import DenseRetriever
from rag_retrieval.retrieval.hybrid import HybridRetriever

VARIANTES = [BM25Retriever, DenseRetriever, HybridRetriever]


def test_retrieved_tiene_campos_esperados():
    r = Retrieved(doc_id="d1", score=1.5, rank=1)
    assert r.doc_id == "d1"
    assert r.score == 1.5
    assert r.rank == 1


def test_retrieved_es_inmutable():
    r = Retrieved(doc_id="d1", score=1.5, rank=1)
    with pytest.raises(Exception):
        r.score = 2.0  # frozen dataclass


def test_retriever_es_abstracto():
    with pytest.raises(TypeError):
        Retriever()  # no se puede instanciar la interfaz directamente


@pytest.mark.parametrize("cls", VARIANTES)
def test_variantes_heredan_la_interfaz(cls):
    assert issubclass(cls, Retriever)
    assert isinstance(cls.name, str) and cls.name


@pytest.mark.parametrize("cls", VARIANTES)
def test_firma_de_retrieve(cls):
    sig = inspect.signature(cls.retrieve)
    assert list(sig.parameters) == ["self", "query", "k"]
    assert sig.parameters["k"].default == 10


@pytest.mark.parametrize("cls", VARIANTES)
def test_stubs_lanzan_not_implemented(cls):
    retriever = cls()
    with pytest.raises(NotImplementedError):
        retriever.index([])
    with pytest.raises(NotImplementedError):
        retriever.retrieve("consulta de prueba", k=5)
