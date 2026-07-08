"""Pruebas del híbrido: fusión RRF con casos calculados a mano."""

from __future__ import annotations

import pytest

from rag_retrieval.retrieval.hybrid import HybridRetriever, rrf_fusion

CORPUS = [
    {"doc_id": "d1", "text": "El perro es un animal doméstico y fiel compañero del ser humano."},
    {"doc_id": "d2", "text": "Las redes neuronales profundas aprenden representaciones jerárquicas."},
    {"doc_id": "d3", "text": "La recuperación de información busca documentos relevantes para una consulta."},
]


def test_rrf_valores_calculados_a_mano():
    # A=[d1,d2,d3], B=[d2,d3,d4], c=60: d2 y d3 aparecen en ambas listas.
    fusion = dict(rrf_fusion([["d1", "d2", "d3"], ["d2", "d3", "d4"]], c=60))
    assert fusion["d2"] == pytest.approx(1 / 62 + 1 / 61)
    assert fusion["d3"] == pytest.approx(1 / 63 + 1 / 62)
    assert fusion["d1"] == pytest.approx(1 / 61)
    assert fusion["d4"] == pytest.approx(1 / 63)


def test_rrf_ordena_por_puntaje_acumulado():
    orden = [doc_id for doc_id, _ in rrf_fusion([["d1", "d2", "d3"], ["d2", "d3", "d4"]])]
    assert orden == ["d2", "d3", "d1", "d4"]


def test_rrf_doc_en_ambas_supera_a_uno_alto_en_una_sola():
    orden = [doc_id for doc_id, _ in rrf_fusion([["d1", "d2"], ["x", "d2"]])]
    assert orden[0] == "d2"


def test_rrf_empate_conserva_primera_aparicion():
    orden = [doc_id for doc_id, _ in rrf_fusion([["d1"], ["d2"]])]
    assert orden == ["d1", "d2"]


def test_hibrido_ranks_consecutivos_desde_uno():
    r = HybridRetriever()
    r.index(CORPUS)
    assert [x.rank for x in r.retrieve("redes neuronales", k=3)] == [1, 2, 3]


def test_hibrido_es_determinista():
    r = HybridRetriever()
    r.index(CORPUS)
    a = r.retrieve("recuperación de documentos", k=3)
    b = r.retrieve("recuperación de documentos", k=3)
    assert [(x.doc_id, x.score) for x in a] == [(x.doc_id, x.score) for x in b]


def test_retrieve_sin_index_lanza_error():
    with pytest.raises(RuntimeError):
        HybridRetriever().retrieve("consulta", k=3)


def test_corpus_vacio_lanza_error():
    with pytest.raises(ValueError):
        HybridRetriever().index([])
