"""Pruebas del recuperador BM25 sobre un mini-corpus con ganadores obvios."""

from __future__ import annotations

import pytest

from rag_retrieval.retrieval.bm25 import BM25Retriever, tokenize

CORPUS = [
    {"doc_id": "d1", "text": "El índice invertido acelera la búsqueda de términos."},
    {"doc_id": "d2", "text": "BM25 es una función de ranking para recuperación de información."},
    {"doc_id": "d3", "text": "Las redes neuronales aprenden representaciones profundas."},
    {"doc_id": "d4", "text": "La similitud coseno compara dos vectores por su ángulo."},
]


@pytest.fixture()
def retriever() -> BM25Retriever:
    r = BM25Retriever()
    r.index(CORPUS)
    return r


def test_tokenize_minusculas_y_sin_tildes():
    assert tokenize("Índice INVERTIDO, búsqueda!") == ["indice", "invertido", "busqueda"]


def test_tokenize_consulta_sin_tildes_coincide_con_corpus():
    assert tokenize("busqueda") == tokenize("búsqueda")


def test_recupera_el_documento_esperado_primero(retriever: BM25Retriever):
    assert retriever.retrieve("función de ranking BM25", k=2)[0].doc_id == "d2"


def test_consulta_con_tildes_distintas(retriever: BM25Retriever):
    assert retriever.retrieve("indice invertido", k=1)[0].doc_id == "d1"


def test_ranks_consecutivos_desde_uno(retriever: BM25Retriever):
    assert [r.rank for r in retriever.retrieve("vectores", k=4)] == [1, 2, 3, 4]


def test_scores_en_orden_descendente(retriever: BM25Retriever):
    scores = [r.score for r in retriever.retrieve("similitud coseno vectores", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_k_mayor_que_el_corpus_no_rompe(retriever: BM25Retriever):
    assert len(retriever.retrieve("bm25", k=50)) == len(CORPUS)


def test_k_limita_resultados(retriever: BM25Retriever):
    assert len(retriever.retrieve("bm25", k=2)) == 2


def test_consulta_sin_terminos_en_comun_da_scores_cero(retriever: BM25Retriever):
    resultados = retriever.retrieve("astronomía galaxias", k=4)
    assert all(r.score == 0.0 for r in resultados)
    # Empate total: se conserva el orden del corpus (determinismo).
    assert [r.doc_id for r in resultados] == ["d1", "d2", "d3", "d4"]


def test_retrieve_sin_index_lanza_error():
    with pytest.raises(RuntimeError):
        BM25Retriever().retrieve("consulta", k=3)


def test_corpus_vacio_lanza_error():
    with pytest.raises(ValueError):
        BM25Retriever().index([])


def test_k_invalido_lanza_error(retriever: BM25Retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("bm25", k=0)
