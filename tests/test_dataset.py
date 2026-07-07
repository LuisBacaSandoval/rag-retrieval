"""Pruebas de integridad del corpus y las consultas: esquema, mínimos y relevancia consistente."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "corpus.jsonl"
QUERIES_PATH = PROJECT_ROOT / "data" / "queries" / "queries.jsonl"

TIPOS_VALIDOS = {"lexica", "semantica", "dificil"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}; ejecuta el pipeline del Hito 2")

    registros = []
    for numero, linea in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        linea = linea.strip()
        if not linea:
            continue
        try:
            registros.append(json.loads(linea))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido en {path.name} línea {numero}") from exc
    return registros


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return load_jsonl(CORPUS_PATH)


@pytest.fixture(scope="module")
def queries() -> list[dict]:
    return load_jsonl(QUERIES_PATH)


# --- Corpus ---


def test_corpus_tamano_minimo(corpus):
    assert len(corpus) >= 100


def test_corpus_esquema(corpus):
    for registro in corpus:
        assert set(registro) >= {"doc_id", "source_id", "title", "text"}
        assert registro["text"].strip()


def test_corpus_doc_id_unicos(corpus):
    ids = [registro["doc_id"] for registro in corpus]
    assert len(ids) == len(set(ids))


# --- Consultas ---


def test_consultas_tamano_minimo(queries):
    assert len(queries) >= 30


def test_consultas_query_id_unicos(queries):
    ids = [q["query_id"] for q in queries]
    assert len(ids) == len(set(ids))


def test_consultas_tipos_validos_y_balanceados(queries):
    conteo = collections.Counter(q["type"] for q in queries)
    assert set(conteo) <= TIPOS_VALIDOS
    for tipo in TIPOS_VALIDOS:
        assert conteo[tipo] >= 8, f"pocas consultas de tipo {tipo}: {conteo[tipo]}"


def test_consultas_tienen_relevancia(queries):
    for q in queries:
        assert q["relevant_source_ids"], f"{q['query_id']} sin fuentes relevantes"


# --- Integridad cruzada ---


def test_relevancia_referencia_fuentes_existentes(corpus, queries):
    fuentes = {registro["source_id"] for registro in corpus}
    for q in queries:
        for source_id in q["relevant_source_ids"]:
            assert source_id in fuentes, (
                f"{q['query_id']} referencia una fuente inexistente: {source_id}"
            )
