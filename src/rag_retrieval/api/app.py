"""API de búsqueda comparativa (FastAPI)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_retrieval.retrieval.base import Retriever
from rag_retrieval.retrieval.bm25 import BM25Retriever
from rag_retrieval.retrieval.dense import DenseRetriever
from rag_retrieval.retrieval.hybrid import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "corpus.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
STATIC_DIR = Path(__file__).resolve().parent / "static"

Metodo = Literal["bm25", "dense", "hybrid"]

EXPLICACION = {
    "bm25": "Léxico: puntúa por coincidencia de términos (IDF, saturación y longitud).",
    "dense": "Denso: similitud coseno entre embeddings multilingües; capta paráfrasis.",
    "hybrid": "Híbrido: fusiona los rankings léxico y denso con Reciprocal Rank Fusion.",
}

app = FastAPI(
    title="Proyecto 5 — Búsqueda comparativa RAG",
    description="Compara recuperación léxica (BM25), densa e híbrida sobre el mismo corpus.",
    version="1.0.0",
)

# Estado cargado una sola vez; se inicializa en el primer uso.
_state: dict = {}


def _load_corpus() -> list[dict]:
    """Lee el corpus de fragmentos desde el JSONL."""
    if not CORPUS_PATH.exists():
        raise RuntimeError(f"No existe el corpus: {CORPUS_PATH}")
    return [
        json.loads(line)
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ensure_loaded() -> dict:
    """Construye/carga los tres recuperadores una única vez."""
    if _state:
        return _state

    corpus = _load_corpus()
    docs = {d["doc_id"]: d for d in corpus}

    bm25 = BM25Retriever()
    bm25.index(corpus)

    # El denso se lee del índice persistido; si falta, se construye en memoria.
    try:
        dense = DenseRetriever.from_persisted(INDEX_DIR)
    except FileNotFoundError:
        dense = DenseRetriever()
        dense.index(corpus)

    hybrid = HybridRetriever(bm25=bm25, dense=dense)

    _state.update(
        corpus=corpus,
        docs=docs,
        retrievers={"bm25": bm25, "dense": dense, "hybrid": hybrid},
    )
    return _state


class Hit(BaseModel):
    rank: int
    doc_id: str
    source_id: str
    title: str
    score: float
    snippet: str


class SearchRequest(BaseModel):
    query: str
    method: Metodo = "hybrid"
    k: int = 5


class SearchResponse(BaseModel):
    method: str
    explicacion: str
    query: str
    k: int
    hits: list[Hit]


def _run(method: Metodo, query: str, k: int) -> list[Hit]:
    """Recupera el top-k con un método y lo enriquece con metadatos del documento."""
    estado = _ensure_loaded()
    retriever: Retriever = estado["retrievers"][method]
    docs = estado["docs"]

    hits: list[Hit] = []
    for r in retriever.retrieve(query, k=k):
        doc = docs[r.doc_id]
        hits.append(
            Hit(
                rank=r.rank,
                doc_id=r.doc_id,
                source_id=doc["source_id"],
                title=doc["title"],
                score=round(r.score, 4),
                snippet=doc["text"][:200].strip() + "…",
            )
        )
    return hits


@app.get("/health")
def health() -> dict:
    """Comprueba que el servicio está activo."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    """Información operativa: tamaño del corpus, métodos y métricas offline si existen."""
    estado = _ensure_loaded()
    info: dict = {
        "fragmentos": len(estado["corpus"]),
        "articulos": len({d["source_id"] for d in estado["corpus"]}),
        "metodos": list(estado["retrievers"]),
        "modelo_denso": estado["retrievers"]["dense"].model_name,
    }
    manifest_path = INDEX_DIR / "dense_manifest.json"
    if manifest_path.exists():
        info["indice_denso"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparacion = PROJECT_ROOT / "results" / "comparacion.csv"
    if comparacion.exists():
        info["evaluacion_offline_csv"] = str(comparacion.relative_to(PROJECT_ROOT))
    return info


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Busca con un único método y devuelve el top-k con su explicación."""
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="La consulta no puede estar vacía")
    if req.k <= 0:
        raise HTTPException(status_code=422, detail="k debe ser mayor que cero")

    return SearchResponse(
        method=req.method,
        explicacion=EXPLICACION[req.method],
        query=req.query,
        k=req.k,
        hits=_run(req.method, req.query, req.k),
    )


@app.get("/compare")
def compare(
    q: str = Query(..., description="Consulta a buscar"),
    k: int = Query(5, gt=0, description="Número de resultados por método"),
) -> dict:
    """Top-k de los tres métodos lado a lado para la misma consulta."""
    if not q.strip():
        raise HTTPException(status_code=422, detail="La consulta no puede estar vacía")

    return {
        "query": q,
        "k": k,
        "resultados": {
            metodo: {
                "explicacion": EXPLICACION[metodo],
                "hits": [h.model_dump() for h in _run(metodo, q, k)],
            }
            for metodo in ("bm25", "dense", "hybrid")
        },
    }


@lru_cache(maxsize=1)
def _index_html() -> str:
    """Carga (y cachea) la página del buscador desde static/index.html."""
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Sirve el buscador HTML comparativo."""
    return _index_html()
