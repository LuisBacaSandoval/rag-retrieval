# Datos: corpus y consultas

_Placeholder — se completa en el Hito 2._

Debe documentar:

- **Origen** del corpus y **licencia**.
- **Tamaño**: número de documentos y de chunks, criterio de chunking.
- **Formato**: `data/corpus/corpus.jsonl` con `{"doc_id", "title", "text"}`.
- **Consultas**: `data/queries/queries.jsonl` con `{"query_id", "query", "type",
  "relevant_doc_ids"}`, 30+ consultas repartidas en tipos (léxicas,
  parafraseadas/semánticas y difíciles).
- **Criterio de relevancia**: cómo se decidió qué documentos son relevantes por
  consulta.
