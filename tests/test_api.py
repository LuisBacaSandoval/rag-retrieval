"""Pruebas de la API (FastAPI) con TestClient: endpoints, validaciones y comparación."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_retrieval.api.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_bm25_devuelve_top_k_ordenado():
    resp = client.post("/search", json={"query": "función de ranking Okapi BM25", "method": "bm25", "k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["method"] == "bm25"
    assert len(data["hits"]) == 3
    assert [h["rank"] for h in data["hits"]] == [1, 2, 3]
    # La consulta léxica exacta debe traer el artículo de BM25 en primer lugar.
    assert data["hits"][0]["source_id"] == "bm25"


def test_search_consulta_vacia_es_422():
    resp = client.post("/search", json={"query": "   ", "method": "bm25", "k": 3})
    assert resp.status_code == 422


def test_search_k_invalido_es_422():
    resp = client.post("/search", json={"query": "bm25", "method": "bm25", "k": 0})
    assert resp.status_code == 422


def test_search_metodo_desconocido_es_422():
    resp = client.post("/search", json={"query": "bm25", "method": "otro", "k": 3})
    assert resp.status_code == 422


def test_metrics_reporta_corpus_y_metodos():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fragmentos"] > 0
    assert set(data["metodos"]) == {"bm25", "dense", "hybrid"}


def test_compare_devuelve_los_tres_metodos():
    resp = client.get("/compare", params={"q": "función de ranking Okapi BM25", "k": 3})
    assert resp.status_code == 200
    resultados = resp.json()["resultados"]
    assert set(resultados) == {"bm25", "dense", "hybrid"}
    assert len(resultados["bm25"]["hits"]) == 3


def test_home_sirve_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Búsqueda comparativa" in resp.text
