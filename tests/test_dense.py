"""Pruebas del recuperador denso (ejecutan el modelo; correr dentro del contenedor)."""

from __future__ import annotations

import pickle

import pytest

from rag_retrieval.retrieval.dense import DenseRetriever

# El denso debe recuperar por significado, no por coincidencia de palabras.
CORPUS = [
    {"doc_id": "d1", "text": "El perro es un animal doméstico y fiel compañero del ser humano."},
    {"doc_id": "d2", "text": "Las redes neuronales profundas aprenden representaciones jerárquicas."},
    {"doc_id": "d3", "text": "La recuperación de información busca documentos relevantes para una consulta."},
]


@pytest.fixture(scope="module")
def retriever() -> DenseRetriever:
    r = DenseRetriever()
    r.index(CORPUS)
    return r


def test_recupera_por_semantica_sin_solapamiento_lexico(retriever: DenseRetriever):
    # "mascota canina" no comparte palabras con d1 ("perro"), pero es su vecino semántico.
    assert retriever.retrieve("mascota canina", k=1)[0].doc_id == "d1"


def test_ranks_consecutivos_desde_uno(retriever: DenseRetriever):
    assert [r.rank for r in retriever.retrieve("aprendizaje automático", k=3)] == [1, 2, 3]


def test_scores_en_orden_descendente(retriever: DenseRetriever):
    scores = [r.score for r in retriever.retrieve("modelos de lenguaje", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_k_mayor_que_el_corpus_no_rompe(retriever: DenseRetriever):
    assert len(retriever.retrieve("perro", k=50)) == len(CORPUS)


def test_determinismo_entre_llamadas(retriever: DenseRetriever):
    primera = retriever.retrieve("redes neuronales", k=3)
    segunda = retriever.retrieve("redes neuronales", k=3)
    assert [(r.doc_id, r.score) for r in primera] == [(r.doc_id, r.score) for r in segunda]


def test_pickle_no_incluye_modelo_ni_faiss_y_sigue_funcionando(retriever: DenseRetriever):
    clon = pickle.loads(pickle.dumps(retriever))
    assert clon.retrieve("mascota canina", k=1)[0].doc_id == "d1"


def test_retrieve_sin_index_lanza_error():
    with pytest.raises(RuntimeError):
        DenseRetriever().retrieve("consulta", k=3)


def test_corpus_vacio_lanza_error():
    with pytest.raises(ValueError):
        DenseRetriever().index([])


def test_k_invalido_lanza_error(retriever: DenseRetriever):
    with pytest.raises(ValueError):
        retriever.retrieve("perro", k=0)
