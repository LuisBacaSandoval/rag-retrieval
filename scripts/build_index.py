"""Construye y persiste el índice denso: ``python scripts/build_index.py``.

Guarda vectores, doc_ids y manifiesto en ``data/indexes/``; patrón de índice con
manifiesto del trabajo de MLOps para RAG (semana 14). BM25 no lo necesita.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_retrieval.retrieval.dense import DenseRetriever, corpus_sha256  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "corpus.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el corpus: {path}")

    corpus = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not corpus:
        raise ValueError(f"El corpus está vacío: {path}")

    return corpus


def main() -> None:
    corpus = load_corpus(CORPUS_PATH)
    corpus_hash = corpus_sha256([doc["text"] for doc in corpus])

    retriever = DenseRetriever()
    retriever.index(corpus)
    manifest = retriever.save(INDEX_DIR, corpus_hash)

    print("Índice denso construido correctamente")
    print(f"Modelo: {manifest['model_name']} (dim {manifest['dim']})")
    print(f"Fragmentos: {manifest['chunks_count']}")
    print(f"Versión del índice: {manifest['index_version']}")
    print(f"Guardado en: {INDEX_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
