"""Consolida los tres métodos: ``python scripts/compare_methods.py``.

Lee ``results/{bm25,dense,hybrid}.json`` y escribe ``comparacion.csv`` (una fila por
método) y ``comparacion_por_consulta.csv`` (RR/recall@10 y efecto de la fusión).
Imprime además los casos notables que alimentan ``results/analisis_errores.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METODOS = {"bm25": "BM25", "dense": "Denso", "hybrid": "Híbrido"}
TIPOS = ["lexica", "semantica", "dificil"]


def load(metodo: str) -> dict:
    # Carga el reporte JSON de un método desde results/.
    return json.loads((RESULTS_DIR / f"{metodo}.json").read_text(encoding="utf-8"))


def tabla_global(reportes: dict[str, dict]) -> pd.DataFrame:
    # Una fila por método: métricas globales, MRR por tipo y costos.
    filas = []
    for clave, nombre in METODOS.items():
        r = reportes[clave]
        fila = {
            "metodo": nombre,
            "mrr": r["global"]["mrr"],
            "precision@5": r["global"]["precision@5"],
            "recall_articulo@5": r["global"]["recall_articulo@5"],
            "ndcg@10": r["global"]["ndcg@10"],
        }
        for t in TIPOS:
            fila[f"mrr_{t}"] = r["por_tipo"][t]["mrr"]
        fila["latencia_p50_ms"] = r["latencia_ms"]["p50"]
        fila["construccion_s"] = r["indice"]["construccion_s"]
        fila["indice_mb"] = r["indice"]["tamano_serializado_mb"]
        filas.append(fila)

    return pd.DataFrame(filas).set_index("metodo")


def tabla_por_consulta(reportes: dict[str, dict]) -> pd.DataFrame:
    # Une RR/recall@10 de los tres métodos por consulta y mide el efecto del híbrido.
    base = None
    for clave in METODOS:
        registros = reportes[clave]["por_consulta"]
        df = pd.DataFrame(
            [
                {
                    "query_id": c["query_id"],
                    "tipo": c["tipo"],
                    "query": c["query"],
                    f"rr_{clave}": round(c["rr"], 3),
                    f"recall10_{clave}": c["recall_articulo@10"],
                }
                for c in registros
            ]
        )
        base = df if base is None else base.merge(df, on=["query_id", "tipo", "query"])

    # Mejor método individual (BM25 o denso) y efecto de la fusión híbrida.
    base["mejor_individual"] = base[["rr_bm25", "rr_dense"]].max(axis=1)
    base["hibrido_vs_individual"] = (base["rr_hybrid"] - base["mejor_individual"]).round(3)
    return base


def main() -> None:
    # Genera los CSV de comparación e imprime los casos notables.
    reportes = {clave: load(clave) for clave in METODOS}

    global_df = tabla_global(reportes)
    global_df.round(3).to_csv(RESULTS_DIR / "comparacion.csv", encoding="utf-8")

    consultas = tabla_por_consulta(reportes)
    consultas.to_csv(RESULTS_DIR / "comparacion_por_consulta.csv", index=False, encoding="utf-8")

    print("== Comparación global ==")
    print(global_df.round(3).to_string())

    print("\n== El híbrido supera a ambos individuales ==")
    mejora = consultas[consultas["hibrido_vs_individual"] > 0]
    print(mejora[["query_id", "tipo", "rr_bm25", "rr_dense", "rr_hybrid"]].to_string(index=False))

    print("\n== La fusión empeora frente al mejor individual ==")
    empeora = consultas[consultas["hibrido_vs_individual"] < 0]
    print(empeora[["query_id", "tipo", "rr_bm25", "rr_dense", "rr_hybrid"]].to_string(index=False))

    print("\n== Los tres fallan por completo (RR = 0) ==")
    fallan = consultas[(consultas[["rr_bm25", "rr_dense", "rr_hybrid"]] == 0).all(axis=1)]
    print(fallan[["query_id", "tipo", "query"]].to_string(index=False))

    print("\n== Léxicas donde el denso pierde frente a BM25 ==")
    lex = consultas[(consultas["tipo"] == "lexica") & (consultas["rr_dense"] < consultas["rr_bm25"])]
    print(lex[["query_id", "rr_bm25", "rr_dense", "rr_hybrid"]].to_string(index=False))

    print(f"\nGuardado: {(RESULTS_DIR / 'comparacion.csv').relative_to(PROJECT_ROOT)}")
    print(f"Guardado: {(RESULTS_DIR / 'comparacion_por_consulta.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
