# Relación con los cuadernos del curso

El proyecto reutiliza patrones del curso CC0C2, con atribución en el encabezado
de cada archivo que lo hace. Mapeo:

| Componente | Origen | Qué se reutilizó / cambió |
|---|---|---|
| `retrieval/bm25.py` | Cuaderno 21 (sem. 10); fórmula explicada según el Cuaderno 23 (sem. 11) | Patrón `rank_bm25`; se agregó tokenización para español y la interfaz común `Retriever` |
| `retrieval/dense.py` | Cuaderno 21 (sem. 10) | Patrón embeddings + FAISS; se cambió a modelo multilingüe y se agregó persistencia con manifiesto |
| `retrieval/hybrid.py` | — (nuevo) | Fusión RRF implementada a mano; no existe en los cuadernos |
| `evaluation/metrics.py` y `run_eval.py` | Proyecto MLOps (sem. 14) y marco de evaluación del Cuaderno 23 (sem. 11) | P@k, R@k y RR de la sem. 14; nDCG nueva; protocolo por tipo de consulta propio |
| API, Docker, scripts y estructura del repo | Proyecto MLOps (sem. 14), principios de la sem. 13 | FastAPI con `/health`/`/metrics`, índice con manifiesto y organización `src/`+`scripts/`+`tests/`; se agregó el endpoint `/compare` y el buscador HTML |

Sobre esa base, lo propio del proyecto es: la variante híbrida con RRF, el
dataset de 54 consultas etiquetadas por tipo, el análisis de errores con casos
trazables en `results/` y la medición de costo (latencia, memoria) junto a la
calidad.
