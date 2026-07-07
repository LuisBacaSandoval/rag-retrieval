# Proyecto 5 — Algoritmos competitivos para recuperación y evaluación en RAG

Proyecto del Examen Final de **CC0C2 — Procesamiento de Lenguaje Natural**.

## Pregunta técnica evaluable

> Comparar recuperación léxica (BM25/TF-IDF), recuperación densa (embeddings) y
> recuperación híbrida (fusión RRF + reranking) sobre un corpus de 100+
> documentos y 30+ consultas, midiendo Precision@k, Recall@k, MRR y nDCG, junto
> con latencia y memoria, y analizando qué tipos de consulta fallan en cada
> método.

## El problema

Un sistema RAG solo responde tan bien como recupera. Este proyecto no asume que
un método sea mejor: **mide** tres estrategias de recuperación sobre el mismo
corpus y las mismas consultas, y estudia *dónde* y *por qué* gana cada una.

- **Línea base — léxica (BM25):** puntúa por coincidencia de términos, frecuencia
  inversa de documento y longitud. Fuerte cuando la consulta usa las mismas
  palabras que el documento.
- **Variante 1 — densa (embeddings):** representa consulta y documentos como
  vectores y mide similitud coseno. Fuerte en paráfrasis y sinónimos, sin
  coincidencia literal.
- **Variante 2 — híbrida (RRF):** fusiona los rankings léxico y denso por
  Reciprocal Rank Fusion para intentar heredar lo mejor de ambos.

## Interfaz común de recuperadores

La decisión de diseño central del proyecto es que **los tres métodos comparten
el mismo contrato**, definido en
[`src/rag_retrieval/retrieval/base.py`](src/rag_retrieval/retrieval/base.py):

```python
class Retriever(ABC):
    def index(self, corpus: list[dict]) -> None: ...
    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]: ...
```

Cada resultado es un `Retrieved(doc_id, score, rank)`. Al recibir la misma
entrada y devolver la misma salida, la evaluación aplica **exactamente el mismo
procedimiento** a BM25, denso e híbrido, sin favorecer a ninguno. El campo
`rank` (posición, no puntaje crudo) es lo que permite la fusión RRF del híbrido,
porque los puntajes de BM25 y del coseno viven en escalas distintas y no son
comparables.

## Estructura del proyecto

```text
proyecto5-rag-retrieval/
  src/rag_retrieval/
    retrieval/   base.py (interfaz), bm25.py, dense.py, hybrid.py
    evaluation/  metrics.py (P@k, R@k, MRR, nDCG)
    api/         app.py (búsqueda comparativa, FastAPI)
  scripts/       prepare_corpus.py, build_index.py, run_eval.py
  data/          corpus/, queries/, indexes/
  notebooks/     proyecto5_rag.ipynb (cuaderno técnico)
  results/       métricas, tablas, análisis de errores
  docs/          datos.md, arquitectura.md
  tests/
```

## Datos

El corpus son **595 fragmentos** de **40 artículos de Wikipedia en español**
sobre PLN, recuperación de información y aprendizaje automático (licencia
CC BY-SA 4.0). Las consultas de evaluación son **54**, etiquetadas en tres tipos
(léxicas, semánticas y difíciles) con juicios de relevancia a nivel de artículo.
Procedencia, licencia, parámetros de chunking y limitaciones en
[`docs/datos.md`](docs/datos.md).

Formatos (una línea JSON por registro):

```text
data/corpus/corpus.jsonl   {"doc_id", "source_id", "title", "text"}
data/queries/queries.jsonl {"query_id", "query", "type", "relevant_source_ids"}
```

La relevancia se etiqueta por artículo (`source_id`); en la evaluación cada
fragmento recuperado se mapea a su artículo de origen. Para regenerar los datos:

```bash
python scripts/fetch_sources.py     # descarga los artículos a data/corpus/raw/
python scripts/prepare_corpus.py    # limpia y trocea -> data/corpus/corpus.jsonl
```

## Evaluación

Cada método se evalúa con el mismo protocolo (54 consultas, k ∈ {1, 3, 5, 10},
latencia y memoria del índice; los detalles del mapeo fragmento→artículo están
en el cuaderno técnico y en el encabezado del script):

```bash
python scripts/run_eval.py --method bm25
```

Deja las métricas globales, por tipo de consulta y por consulta en
`results/bm25.json`, y una tabla plana en `results/bm25_por_consulta.csv`.

## Entorno y reproducción

El proyecto se desarrolla y valida sobre **Python 3.11**. Todas las
dependencias están fijadas en [`requirements.txt`](requirements.txt)
(`rank-bm25`, `sentence-transformers`, `faiss-cpu`, `fastapi`, etc.).

### Opción A — entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
PYTHONPATH=src pytest -q        # pruebas
```

### Opción B — Cuaderno técnico con Docker (recomendado)

Con [`docker-compose.yml`](docker-compose.yml) el proyecto levanta su propio
JupyterLab, sin depender de ningún entorno externo:

```bash
docker compose up notebook
```

En los registros de la consola aparece una URL con token
(`http://127.0.0.1:8888/lab?token=...`); ábrela en el navegador, entra a
`notebooks/`, abre `proyecto5_rag.ipynb` y ejecuta las celdas. El proyecto se
monta dentro del contenedor, así que las salidas y los gráficos se guardan en el
repositorio. La primera construcción tarda (descarga `torch` vía
`sentence-transformers`); las siguientes son inmediatas.

### Opción C — API con Docker

El [`Dockerfile`](Dockerfile) sirve la API de despliegue (Hito 7):

```bash
docker compose up api          # -> http://localhost:8000
# o directamente:
docker build -t proyecto5-rag .
docker run --rm -p 8000:8000 proyecto5-rag
```

## Estado por hitos

| Hito | Contenido | Estado |
|---|---|---|
| 1 | Estructura + interfaz común de recuperadores | ✅ |
| 2 | Corpus (595 fragmentos) y 54 consultas etiquetadas | ✅ |
| 3 | Línea base BM25 + métricas (P@k, R@k, MRR, nDCG) | ✅ |
| 4 | Recuperador denso + índice | pendiente |
| 5 | Recuperador híbrido (RRF) | pendiente |
| 6 | Evaluación comparativa y análisis de errores | pendiente |
| 7 | API de búsqueda comparativa + despliegue | pendiente |
| 8 | Documentación final + video | pendiente |

## Atribución y reutilización

Este proyecto adopta el patrón probado en el curso **CC0C2** durante la
**semana 14** (trabajo propio del autor sobre MLOps para RAG): la estructura
`src/` + `scripts/` + `tests/`, la idea de índice con manifiesto reproducible,
las métricas de recuperación base y el patrón de API con FastAPI. Los archivos
que reutilizan ese patrón lo indican en su encabezado.

## Video

Enlace al video (> 10 min): _pendiente (Hito 8)_.
