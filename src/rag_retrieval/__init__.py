"""Proyecto 5 — Algoritmos competitivos para recuperación y evaluación en RAG.

Compara recuperación léxica (BM25), densa (embeddings) e híbrida (RRF) sobre un
mismo corpus y conjunto de consultas, midiendo P@k, R@k, MRR, nDCG, latencia y
memoria. Todos los métodos comparten la interfaz común definida en
``rag_retrieval.retrieval.base`` para que la comparación sea justa.
"""

__version__ = "0.1.0"
