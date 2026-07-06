"""Prepara el corpus y las consultas de evaluación (Hito 2).

Pendiente de implementación. Limpiará y trocear los documentos en chunks
(~300-500 tokens), guardará ``data/corpus/corpus.jsonl`` con
``{"doc_id", "title", "text"}`` y ``data/queries/queries.jsonl`` con
``{"query_id", "query", "type", "relevant_doc_ids"}``.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Se implementa en el Hito 2 (corpus y consultas).")


if __name__ == "__main__":
    main()
