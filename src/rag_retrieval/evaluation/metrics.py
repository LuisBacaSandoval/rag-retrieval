"""Métricas de recuperación a mano (solo ``math``).

P@k, R@k y RR adaptadas del trabajo de MLOps de la semana 14; nDCG nueva.
Fórmulas e interpretación en el cuaderno técnico, sección 2.2.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fracción de los primeros ``k`` recuperados que son relevantes (divide entre ``k`` fijo)."""
    if k <= 0:
        raise ValueError("k debe ser mayor que cero")

    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fracción de los relevantes que aparecen entre los primeros ``k``."""
    if k <= 0:
        raise ValueError("k debe ser mayor que cero")

    if not relevant:
        return 0.0

    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Inverso de la posición del primer relevante (0 si no aparece); su promedio es el MRR."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """DCG con relevancia binaria normalizado por el ranking ideal (relevantes primero)."""
    if k <= 0:
        raise ValueError("k debe ser mayor que cero")

    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )

    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg
