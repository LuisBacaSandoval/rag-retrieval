"""Interfaz común de recuperadores.

Esta es la decisión de diseño central del proyecto: todos los métodos de
recuperación (léxico BM25, denso por embeddings, híbrido con fusión RRF)
implementan el mismo contrato ``Retriever``, con la misma entrada (una consulta
y un ``k``) y la misma salida (una lista ordenada de ``Retrieved``). Gracias a
esto, la evaluación (``rag_retrieval.evaluation``) y la comparación de métricas
se hacen exactamente igual para los tres métodos, sin favorecer a ninguno.

Se generaliza el patrón de recuperación (dato ``RetrievedChunk`` y función
``retrieve()``) del trabajo de MLOps para RAG del curso CC0C2 (semana 14),
convirtiéndolo en una interfaz abstracta y añadiendo el campo ``rank``, que la
fusión Reciprocal Rank Fusion (RRF) del híbrido necesita para operar sobre
posiciones en lugar de puntajes crudos (que no son comparables entre métodos).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Retrieved:
    """Un documento recuperado para una consulta.

    Attributes:
        doc_id: Identificador del documento o *chunk* dentro del corpus.
        score: Puntaje asignado por el método. Sirve para ordenar dentro de un
            mismo método, pero **no es comparable entre métodos** (BM25 y coseno
            viven en escalas distintas).
        rank: Posición en el ranking, empezando en 1. Sí es comparable entre
            métodos y es lo que usa la fusión RRF del recuperador híbrido.
    """

    doc_id: str
    score: float
    rank: int


class Retriever(ABC):
    """Contrato común de recuperación.

    Toda variante (BM25, densa, híbrida) hereda de esta clase. El flujo de uso
    es siempre el mismo::

        retriever = AlgunRetriever()
        retriever.index(corpus)                 # una vez
        resultados = retriever.retrieve(q, k=10)  # por consulta

    donde ``corpus`` es una lista de diccionarios con al menos las claves
    ``doc_id`` y ``text``.
    """

    #: Nombre corto del método, usado para etiquetar resultados y métricas.
    name: str = "base"

    @abstractmethod
    def index(self, corpus: list[dict]) -> None:
        """Construye el índice interno a partir del corpus.

        Args:
            corpus: Lista de documentos, cada uno como ``{"doc_id": str,
                "text": str, ...}``. Claves adicionales (título, metadatos) se
                ignoran salvo que el método las use.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]:
        """Recupera los ``k`` documentos más relevantes para la consulta.

        Args:
            query: Consulta en lenguaje natural.
            k: Número máximo de resultados a devolver.

        Returns:
            Lista de ``Retrieved`` ordenada de mayor a menor relevancia, con
            ``rank`` de 1 a ``k`` (o menos si el corpus es más pequeño).
        """
        raise NotImplementedError
