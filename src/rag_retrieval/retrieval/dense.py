"""Recuperador denso por embeddings (variante propuesta 1).

Adapta el patrón embeddings + FAISS del Cuaderno 21 (semana 10) a un modelo
multilingüe; teoría en el cuaderno técnico, sección 2.3.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_retrieval.retrieval.base import Retrieved, Retriever

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class DenseRetriever(Retriever):
    """Recuperación densa por similitud coseno entre embeddings."""

    name = "dense"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._faiss: faiss.Index | None = None
        self._embeddings: np.ndarray | None = None
        self._doc_ids: list[str] = []
        self._dim: int | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Embeddings normalizados en float32, listos para ``IndexFlatIP`` (coseno)."""
        vectors = self._load_model().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.ascontiguousarray(vectors, dtype="float32")

    def _build_faiss(self) -> None:
        assert self._embeddings is not None
        self._dim = int(self._embeddings.shape[1])
        self._faiss = faiss.IndexFlatIP(self._dim)
        self._faiss.add(self._embeddings)

    def index(self, corpus: list[dict]) -> None:  # noqa: D102 (ver Retriever)
        if not corpus:
            raise ValueError("El corpus no puede estar vacío")

        self._doc_ids = [doc["doc_id"] for doc in corpus]
        self._embeddings = self._encode([doc["text"] for doc in corpus])
        self._build_faiss()

    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:  # noqa: D102
        if self._faiss is None:
            raise RuntimeError("Hay que llamar a index() antes de retrieve()")

        if k <= 0:
            raise ValueError("k debe ser mayor que cero")

        query_vector = self._encode([query])
        limit = min(k, len(self._doc_ids))
        scores, indices = self._faiss.search(query_vector, limit)

        # Desempate estable como en BM25: ante igual puntaje gana el menor índice.
        pares = sorted(
            zip(indices[0].tolist(), scores[0].tolist()),
            key=lambda p: (-p[1], p[0]),
        )
        return [
            Retrieved(doc_id=self._doc_ids[i], score=float(s), rank=rank)
            for rank, (i, s) in enumerate(pares, start=1)
        ]

    def manifest(self, corpus_hash: str) -> dict:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_name": self.model_name,
            "dim": self._dim,
            "chunks_count": len(self._doc_ids),
            "embeddings_bytes": int(self._embeddings.nbytes) if self._embeddings is not None else 0,
            "corpus_hash_sha256": corpus_hash,
            "index_version": corpus_hash[:12],
        }

    def save(self, index_dir: Path, corpus_hash: str) -> dict:
        """Persiste embeddings, doc_ids y manifiesto en ``index_dir``."""
        if self._embeddings is None:
            raise RuntimeError("Hay que llamar a index() antes de save()")

        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "dense_embeddings.npy", self._embeddings)
        (index_dir / "dense_doc_ids.json").write_text(
            json.dumps(self._doc_ids, ensure_ascii=False), encoding="utf-8"
        )
        manifest = self.manifest(corpus_hash)
        (index_dir / "dense_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return manifest

    @classmethod
    def from_persisted(cls, index_dir: Path) -> DenseRetriever:
        """Carga el índice del disco sin recodificar el corpus (arranque rápido)."""
        manifest = json.loads((index_dir / "dense_manifest.json").read_text(encoding="utf-8"))
        retriever = cls(model_name=manifest["model_name"])
        retriever._embeddings = np.load(index_dir / "dense_embeddings.npy")
        retriever._doc_ids = json.loads(
            (index_dir / "dense_doc_ids.json").read_text(encoding="utf-8")
        )
        retriever._build_faiss()
        return retriever

    # El modelo y el índice FAISS (objetos SWIG) no son picklables; se reconstruyen
    # desde los embeddings, que son el verdadero costo del índice.
    def __getstate__(self) -> dict:
        estado = self.__dict__.copy()
        estado["_model"] = None
        estado["_faiss"] = None
        return estado

    def __setstate__(self, estado: dict) -> None:
        self.__dict__.update(estado)
        if self._embeddings is not None:
            self._build_faiss()


def corpus_sha256(texts: list[str]) -> str:
    """Hash del contenido del corpus para versionar el índice (patrón semana 14)."""
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
