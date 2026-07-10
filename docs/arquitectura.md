# Arquitectura

Explicación técnica del pipeline, de cada algoritmo de recuperación y de las
decisiones de diseño. Los datos se describen en [`datos.md`](datos.md), las
limitaciones en [`limitaciones.md`](limitaciones.md) y la relación con los
cuadernos del curso en [`relacion_cuadernos.md`](relacion_cuadernos.md).

## 1. Visión general del pipeline

```text
                         PREPARACIÓN (offline, reproducible)
  Wikipedia ES ──fetch_sources.py──> data/corpus/raw/ (40 artículos con procedencia)
                                          │
                                 prepare_corpus.py (limpieza + chunking 800/120)
                                          │
                                          v
              data/corpus/corpus.jsonl (595 fragmentos)     data/queries/queries.jsonl
                                          │                  (54 consultas etiquetadas)
                                          │                             │
                         INDEXACIÓN      │                             │
             ┌────────────────────┬──────┴───────────┐                 │
             v                    v                  v                 │
        BM25Retriever       DenseRetriever     HybridRetriever         │
        (frecuencias        (embeddings 384d   (compone los otros      │
         en memoria)         + FAISS FlatIP)    dos, fusión RRF)       │
             │                    │                  │                 │
             └────────────────────┴──────┬───────────┘                 │
                                         v                             v
                     interfaz común  retrieve(query, k)          run_eval.py
                     -> list[Retrieved(doc_id, score, rank)]   (protocolo idéntico
                                         │                      para los 3 métodos)
                          ┌──────────────┴──────────────┐              │
                          v                             v              v
                   api/app.py (FastAPI)          results/<metodo>.json + _por_consulta.csv
                   /  /search  /compare                 │
                   /health  /metrics             compare_methods.py
                          │                             │
                          v                             v
                   Docker (docker               results/comparacion*.csv
                   compose up api)              results/analisis_errores.md
```

Dos propiedades sostienen todo el diseño:

1. **Separación datos / índice / evaluación / servicio.** Cada etapa deja un
   artefacto versionable (JSONL, `.npy` + manifiesto, JSON/CSV de métricas), de
   modo que cualquier resultado del README o del cuaderno puede regenerarse con
   un script concreto.
2. **Interfaz común de recuperadores.** La comparación es justa porque los tres
   métodos reciben la misma entrada y devuelven la misma salida; la evaluación
   no sabe (ni necesita saber) qué método está midiendo.

## 2. La interfaz común `Retriever`

Definida en [`src/rag_retrieval/retrieval/base.py`](../src/rag_retrieval/retrieval/base.py):

```python
class Retriever(ABC):
    def index(self, corpus: list[dict]) -> None: ...
    def retrieve(self, query: str, k: int = 10) -> list[Retrieved]: ...

@dataclass(frozen=True)
class Retrieved:
    doc_id: str
    score: float   # puntaje del método (NO comparable entre métodos)
    rank: int      # posición 1..k (SÍ comparable; la usa RRF)
```

La distinción `score`/`rank` es deliberada: un puntaje BM25 (escala abierta,
depende de frecuencias del corpus) y una similitud coseno (acotada en [-1, 1])
viven en escalas distintas y no pueden sumarse ni promediarse directamente. La
**posición** en el ranking, en cambio, es adimensional y comparable, y es lo que
permite la fusión híbrida por RRF sin normalizaciones ad hoc.

## 3. Los tres algoritmos

### 3.1 BM25 (línea base léxica)

[`bm25.py`](../src/rag_retrieval/retrieval/bm25.py) — Okapi BM25 vía `rank_bm25`.

Puntúa por coincidencia de términos, ponderando cada uno por su rareza en el
corpus (IDF), con saturación de la frecuencia (`k1 = 1.5`) y normalización por
longitud del documento (`b = 0.75`). La fórmula desarrollada e interpretada
está en el cuaderno técnico, sección 2.1.

El "índice" son las frecuencias de término en memoria (~0.6 MB para los 595
fragmentos). La tokenización (minúsculas, sin tildes, `\w+`) es **compartida**
entre corpus y consultas y está centralizada en `bm25.tokenize` — si consulta y
documento se tokenizaran distinto, la coincidencia léxica se rompería
silenciosamente. El desempate del ranking es estable (a igual puntaje gana el
documento que aparece antes en el corpus), lo que hace la evaluación
determinista.

- Costo: construcción ~0.15 s; consulta < 1 ms (p50 0.7 ms). Es la referencia
  de costo mínimo contra la que se juzgan las variantes.

### 3.2 Denso (variante 1: embeddings)

[`dense.py`](../src/rag_retrieval/retrieval/dense.py) — `sentence-transformers`
+ FAISS.

Cada fragmento y cada consulta se codifican con
`paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones, multilingüe, corre en
CPU) a un vector que representa su **significado**, y se comparan por
**similitud coseno**: textos parafraseados quedan cerca aunque no compartan
palabras (teoría en el cuaderno, sección 3.1). Como los embeddings se
normalizan a norma 1, el coseno equivale al producto interno, y el índice es un
`faiss.IndexFlatIP`: búsqueda **exacta** (fuerza bruta sobre los 595 vectores),
no aproximada — a esta escala no se necesita un índice ANN y se elimina una
fuente de error.

Persistencia y reproducibilidad ([`scripts/build_index.py`](../scripts/build_index.py)):
los embeddings se guardan en `data/indexes/dense_embeddings.npy` junto con un
**manifiesto** (modelo, dimensión, nº de fragmentos, hash SHA-256 del corpus,
versión del índice). Si el corpus cambia, el hash delata que el índice quedó
obsoleto. La API arranca con `DenseRetriever.from_persisted()` sin recodificar
el corpus.

- Costo: construcción ~18 s (codificar 595 fragmentos); consulta p50 ~15 ms,
  dominada por codificar la consulta con el transformer (la búsqueda FAISS en sí
  es sub-milisegundo). Índice ~0.9 MB.

### 3.3 Híbrido (variante 2: fusión RRF)

[`hybrid.py`](../src/rag_retrieval/retrieval/hybrid.py) — Reciprocal Rank Fusion
de BM25 + denso.

Cada método aporta sus `pool = 50` mejores candidatos y cada documento acumula
`1/(c + posición)` por cada lista en la que aparece (fórmula e interpretación en
el cuaderno, sección 4.1). Fusiona por **posiciones y no por puntajes** porque
los puntajes no son comparables entre métodos (ver §2). Sus dos hiperparámetros:

- **`c = 60`** amortigua el peso de los primeros puestos: aparecer
  razonablemente arriba en *ambas* listas vale más que ser #1 en una sola. Es el
  valor del paper original de RRF (Cormack et al., 2009) y se mantuvo sin ajustar.
- **`pool = 50`** da margen para que un documento profundo en una lista y medio
  en la otra entre al top-k final; la fusión solo puede promover lo que los
  métodos base recuperan.

El híbrido compone las dos implementaciones anteriores sin duplicar código:
indexa ambas y fusiona posiciones. Su costo es aproximadamente la suma de los
dos (p50 ~16 ms, índice ~1.5 MB).

Este mecanismo explica también sus dos modos de fallo medidos (detalle en
[`results/analisis_errores.md`](../results/analisis_errores.md)): **dilución**
(un hallazgo profundo de un solo método se pierde al sumarse con el cero del
otro) y **consenso sobre especialización** (un documento mediocre en ambas
listas supera al que era #1 en una sola).

## 4. Evaluación

[`metrics.py`](../src/rag_retrieval/evaluation/metrics.py) implementa P@k, R@k,
RR y nDCG a mano (solo `math`), verificadas en `tests/test_metrics.py` con casos
calculados manualmente. El protocolo ([`scripts/run_eval.py`](../scripts/run_eval.py))
es idéntico para los tres métodos:

| Elemento | Decisión |
|---|---|
| Consultas | Las 54, por tipo (léxica / semántica / difícil) |
| Cortes | k ∈ {1, 3, 5, 10} |
| Relevancia | Etiquetada por **artículo**; cada fragmento recuperado hereda la relevancia de su `source_id` |
| P@k, MRR, nDCG | A nivel de **fragmento** (miden el ranking que ve el usuario) |
| Recall@k | A nivel de **artículo** (mide cobertura: cuántas fuentes relevantes aparecen) |
| Latencia | `perf_counter` con calentamiento previo; se reporta la **mediana (p50)**, robusta a picos del entorno |
| Memoria | `tracemalloc` durante la construcción + tamaño serializado (pickle) del índice |

Las métricas de calidad son **deterministas** (mismo corpus, mismas consultas,
desempates estables); solo la latencia varía entre corridas y máquinas.

## 5. API y despliegue

[`api/app.py`](../src/rag_retrieval/api/app.py) sirve los tres recuperadores
tras la misma interfaz: `/` (buscador HTML comparativo), `POST /search`,
`GET /compare` (los tres métodos lado a lado — el requisito de "comparación
visible"), `/health` y `/metrics` (expone el manifiesto del índice). El estado
se carga una sola vez al primer uso: el denso desde el índice persistido, BM25 y
el híbrido reindexando en memoria (~0.2 s).

Docker define dos servicios independientes en
[`docker-compose.yml`](../docker-compose.yml):

- `notebook` — JupyterLab con las dependencias fijadas, para ejecutar el
  cuaderno técnico de forma reproducible.
- `api` — imagen que construye el índice denso en build y sirve la API con
  `uvicorn` (evidencia en
  [`results/evidencia_despliegue/`](../results/evidencia_despliegue/)).

## 6. Resumen de decisiones de diseño

| Decisión | Valor | Por qué |
|---|---|---|
| Chunking | 800 caracteres, solape 120 | Fragmentos autocontenidos sin partir ideas en el borde; ver `datos.md` |
| Relevancia | Por artículo, binaria | Etiquetar 595×54 fragmentos es inviable y frágil ante re-chunking |
| Tokenización léxica | minúsculas + sin tildes + `\w+`, compartida | Coherencia consulta/corpus; el español pierde recall si las tildes distinguen tokens |
| Modelo denso | MiniLM multilingüe 384d, CPU | Corpus en español; ligero y reproducible sin GPU |
| Índice denso | FAISS `IndexFlatIP` exacto | 595 vectores no justifican ANN; elimina el error de aproximación |
| Fusión | RRF con c=60, pool=50 | Valores del paper original; rangos en vez de puntajes no comparables |
| Latencia | Mediana (p50) | Robusta ante picos puntuales del entorno |
| Desempates | Estables en los tres métodos | Evaluación determinista y reproducible |
