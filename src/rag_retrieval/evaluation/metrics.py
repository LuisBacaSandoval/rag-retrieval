"""Métricas de evaluación de recuperación.

Pendiente de implementación en el Hito 3. Las tres primeras (Precision@k,
Recall@k, Reciprocal Rank) reutilizan la implementación propia del trabajo de
MLOps para RAG del curso CC0C2 (semana 14); en el Hito 3 se añade nDCG@k. Se
implementan a mano, sin librerías de métricas, para tener control total del
cálculo.
"""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fracción de los primeros ``k`` recuperados que son relevantes."""
    raise NotImplementedError("Se implementa en el Hito 3 (métricas).")


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fracción de los relevantes que aparecen entre los primeros ``k``."""
    raise NotImplementedError("Se implementa en el Hito 3 (métricas).")


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Inverso de la posición del primer documento relevante (0 si no aparece)."""
    raise NotImplementedError("Se implementa en el Hito 3 (métricas).")


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Ganancia acumulada descontada normalizada en los primeros ``k``."""
    raise NotImplementedError("Se implementa en el Hito 3 (métricas).")
