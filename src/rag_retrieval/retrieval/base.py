"""Interfaz base para todos los recuperadores (BM25, denso e híbrido).

Todos implementan el mismo contrato (`index` y `retrieve`), lo que permite
evaluarlos y compararlos de forma uniforme.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Retrieved:
    """Documento recuperado con su puntaje y posición en el ranking."""

    doc_id: str
    score: float  # Puntaje del método (no comparable entre métodos).
    rank: int  # Posición del documento (1..k), usada por RRF.


class Retriever(ABC):
    """Interfaz común para los métodos de recuperación."""

    #: Nombre del recuperador.
    name: str = "base"

    @abstractmethod
    def index(self, corpus: list[dict]) -> None:
        """Construye el índice a partir del corpus."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:
        """Devuelve los `k` documentos más relevantes para la consulta."""
        raise NotImplementedError