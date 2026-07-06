# Arquitectura y limitaciones

_Placeholder — se completa en el Hito 8._

Debe incluir:

- **Diagrama del pipeline**: corpus -> índice(s) -> recuperador -> evaluación / API.
- **Interfaz común** `Retriever` y por qué habilita una comparación justa.
- **Cada algoritmo** explicado: cómo pondera BM25, qué representa un embedding y
  la similitud coseno, cómo fusiona RRF por posiciones.
- **Decisiones** de diseño (chunking, modelo de embeddings, k, etc.).
- **Limitaciones**: corpus pequeño, un solo modelo de embeddings, juicios de
  relevancia de una sola persona, etc.
- **Relación con los cuadernos del curso** (embeddings, búsqueda semántica, RAG,
  evaluación de retrieval, MLOps).
