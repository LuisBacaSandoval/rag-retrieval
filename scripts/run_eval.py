"""Evalúa un recuperador sobre las 54 consultas: ``python scripts/run_eval.py --method bm25``.

Guarda métricas globales, por tipo y por consulta en ``results/<metodo>.json`` y
``results/<metodo>_por_consulta.csv``. P@k/nDCG/MRR se miden a nivel de fragmento
(relevancia heredada del artículo) y Recall@k a nivel de artículo; detalle en el
cuaderno, sección 2.2. Estructura adaptada de ``evaluate_rag.py`` (semana 14).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_retrieval.evaluation.metrics import (  # noqa: E402
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_retrieval.retrieval.base import Retriever  # noqa: E402
from rag_retrieval.retrieval.bm25 import BM25Retriever  # noqa: E402
from rag_retrieval.retrieval.dense import DenseRetriever  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "corpus.jsonl"
QUERIES_PATH = PROJECT_ROOT / "data" / "queries" / "queries.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"

KS_DEFAULT = [1, 3, 5, 10]


def config_del_metodo(retriever: Retriever, ks: list[int]) -> dict[str, Any]:
    """Configuración específica de cada método para el reporte (sin campos ajenos)."""
    config: dict[str, Any] = {"ks": ks}
    if isinstance(retriever, BM25Retriever):
        config["k1"] = retriever.k1
        config["b"] = retriever.b
        config["tokenizador"] = "minúsculas + sin tildes + \\w+"
    elif isinstance(retriever, DenseRetriever):
        config["model_name"] = retriever.model_name
        config["dim"] = retriever._dim
        config["similitud"] = "coseno (IndexFlatIP sobre vectores normalizados)"

    return config


def build_retriever(method: str) -> Retriever:
    """Instancia el método pedido; el híbrido se agrega en el Hito 4, parte 2."""
    if method == "bm25":
        return BM25Retriever()
    if method == "dense":
        return DenseRetriever()

    raise SystemExit(
        f"Método desconocido o aún no implementado: {method!r} (disponibles: bm25, dense)"
    )


def load_jsonl(path: Path, required: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        row = json.loads(line)
        missing = required - set(row)
        if missing:
            raise ValueError(f"Faltan campos {sorted(missing)} en {path.name}:{line_number}")

        rows.append(row)

    if not rows:
        raise ValueError(f"El archivo está vacío: {path}")

    return rows


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def percentile(values: list[float], q: float) -> float:
    """Percentil por interpolación lineal (q en [0, 1])."""
    if not values:
        return 0.0

    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def evaluate_query(
    query: dict[str, Any],
    retriever: Retriever,
    source_of: dict[str, str],
    relevant_chunks_of: dict[str, set[str]],
    ks: list[int],
) -> dict[str, Any]:
    """Recupera una vez con el k máximo y calcula las métricas en cada corte."""
    relevant_sources = set(query["relevant_source_ids"])
    relevant_chunks = set().union(
        *(relevant_chunks_of.get(source, set()) for source in relevant_sources)
    )

    start = time.perf_counter()
    results = retriever.retrieve(str(query["query"]), k=max(ks))
    latency_ms = (time.perf_counter() - start) * 1000.0

    retrieved_chunks = [r.doc_id for r in results]

    record: dict[str, Any] = {
        "query_id": query["query_id"],
        "tipo": query["type"],
        "query": query["query"],
        "relevant_source_ids": sorted(relevant_sources),
        "latencia_ms": round(latency_ms, 3),
        "rr": reciprocal_rank(retrieved_chunks, relevant_chunks),
        "top": [
            {
                "rank": r.rank,
                "doc_id": r.doc_id,
                "source_id": source_of[r.doc_id],
                "score": round(r.score, 4),
            }
            for r in results
        ],
    }

    for k in ks:
        sources_at_k = unique_in_order([source_of[d] for d in retrieved_chunks[:k]])
        record[f"precision@{k}"] = precision_at_k(retrieved_chunks, relevant_chunks, k)
        record[f"recall_articulo@{k}"] = recall_at_k(sources_at_k, relevant_sources, k)
        record[f"ndcg@{k}"] = ndcg_at_k(retrieved_chunks, relevant_chunks, k)

    return record


def aggregate(records: list[dict[str, Any]], ks: list[int]) -> dict[str, float]:
    summary = {"consultas": len(records), "mrr": round(mean([r["rr"] for r in records]), 4)}
    for k in ks:
        for metric in (f"precision@{k}", f"recall_articulo@{k}", f"ndcg@{k}"):
            summary[metric] = round(mean([r[metric] for r in records]), 4)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="bm25", help="Método a evaluar (bm25)")
    parser.add_argument(
        "--k", nargs="+", type=int, default=KS_DEFAULT, help="Cortes k a reportar"
    )
    args = parser.parse_args()

    ks = sorted(set(args.k))
    if any(k <= 0 for k in ks):
        raise SystemExit("Todos los k deben ser mayores que cero")

    corpus = load_jsonl(CORPUS_PATH, required={"doc_id", "source_id", "text"})
    queries = load_jsonl(
        QUERIES_PATH, required={"query_id", "query", "type", "relevant_source_ids"}
    )

    # Mapeos fragmento <-> artículo para los dos niveles de evaluación.
    source_of = {doc["doc_id"]: doc["source_id"] for doc in corpus}
    relevant_chunks_of: dict[str, set[str]] = {}
    for doc in corpus:
        relevant_chunks_of.setdefault(doc["source_id"], set()).add(doc["doc_id"])

    retriever = build_retriever(args.method)

    # Costo del índice: tiempo de construcción y pico de memoria.
    tracemalloc.start()
    build_start = time.perf_counter()
    retriever.index(corpus)
    build_seconds = time.perf_counter() - build_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    index_pickle_bytes = len(pickle.dumps(retriever))

    # Consulta de calentamiento: la primera llamada paga costos únicos.
    retriever.retrieve(str(queries[0]["query"]), k=max(ks))

    records = [
        evaluate_query(query, retriever, source_of, relevant_chunks_of, ks)
        for query in queries
    ]

    latencies = [r["latencia_ms"] for r in records]
    tipos = sorted({r["tipo"] for r in records})

    report = {
        "metodo": retriever.name,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config_del_metodo(retriever, ks),
        "corpus": {
            "fragmentos": len(corpus),
            "articulos": len(relevant_chunks_of),
        },
        "indice": {
            "construccion_s": round(build_seconds, 3),
            "memoria_pico_construccion_mb": round(peak_bytes / 2**20, 2),
            "tamano_serializado_mb": round(index_pickle_bytes / 2**20, 2),
        },
        "latencia_ms": {
            "media": round(mean(latencies), 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
        },
        "global": aggregate(records, ks),
        "por_tipo": {
            tipo: aggregate([r for r in records if r["tipo"] == tipo], ks)
            for tipo in tipos
        },
        "por_consulta": records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / f"{retriever.name}.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Tabla plana por consulta (sin el top recuperado) para inspección rápida.
    import pandas as pd

    csv_path = RESULTS_DIR / f"{retriever.name}_por_consulta.csv"
    table = pd.DataFrame(
        [{key: value for key, value in r.items() if key != "top"} for r in records]
    )
    table["relevant_source_ids"] = table["relevant_source_ids"].apply("|".join)
    table.to_csv(csv_path, index=False, encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "por_consulta"},
                     indent=2, ensure_ascii=False))
    print(f"\nGuardado: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Guardado: {csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
