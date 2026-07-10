"""Descarga los artículos fuente a ``data/corpus/raw/`` desde Wikipedia en español.

Solo stdlib; ``raw/`` se versiona, así que solo hace falta para regenerar el corpus.
Uso: ``python scripts/fetch_sources.py``.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "corpus" / "raw"

WIKI_LANG = "es"
API_URL = f"https://{WIKI_LANG}.wikipedia.org/w/api.php"
USER_AGENT = "proyecto5-rag-retrieval/0.1 (corpus academico CC0C2)"

# source_id -> título exacto del artículo en Wikipedia en español.
ARTICLES: dict[str, str] = {
    "pln": "Procesamiento de lenguajes naturales",
    "recuperacion_informacion": "Recuperación de información",
    "bm25": "Okapi BM25",
    "tfidf": "Tf-idf",
    "word_embedding": "Word embedding",
    "word2vec": "Word2vec",
    "modelo_lenguaje": "Modelo de lenguaje",
    "transformador": "Transformador (modelo de aprendizaje automático)",
    "atencion": "Atención (aprendizaje automático)",
    "busqueda_semantica": "Búsqueda semántica",
    "similitud_coseno": "Similitud coseno",
    "indice_invertido": "Índice invertido",
    "precision_exhaustividad": "Precisión y exhaustividad",
    "ngrama": "N-grama",
    "ner": "Reconocimiento de entidades nombradas",
    "analisis_sentimiento": "Análisis de sentimiento",
    "traduccion_automatica": "Traducción automática",
    "motor_busqueda": "Motor de búsqueda",
    "aprendizaje_automatico": "Aprendizaje automático",
    "red_neuronal": "Red neuronal artificial",
    "lematizacion": "Lematización",
    "stemming": "Stemming",
    "modelo_espacio_vectorial": "Modelo de espacio vectorial",
    "palabra_vacia": "Palabra vacía",
    "etiquetado_gramatical": "Etiquetado gramatical",
    "linguistica_computacional": "Lingüística computacional",
    "mineria_textos": "Minería de textos",
    "reconocimiento_habla": "Reconocimiento del habla",
    "ley_zipf": "Ley de Zipf",
    "aprendizaje_profundo": "Aprendizaje profundo",
    "red_neuronal_recurrente": "Red neuronal recurrente",
    "perceptron_multicapa": "Perceptrón multicapa",
    "retropropagacion": "Propagación hacia atrás",
    "bert": "BERT (modelo de lenguaje)",
    "svm": "Máquina de vectores de soporte",
    "naive_bayes": "Clasificador bayesiano ingenuo",
    "hmm": "Modelo oculto de Márkov",
    "analisis_grupos": "Análisis de grupos",
    "sistema_recomendacion": "Sistema de recomendación",
    "entropia": "Entropía (información)",
}


def fetch_extract(title: str, max_retries: int = 5) -> tuple[str, str]:
    # Devuelve (texto_plano, url) del extracto, reintentando ante HTTP 429.
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": "1",
        "exsectionformat": "plain",
        "inprop": "url",
        "redirects": "1",
        "titles": title,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries - 1:
                espera = 5 * (attempt + 1)
                print(f"[429] límite de tasa; reintento en {espera}s ({title})")
                time.sleep(espera)
                continue
            raise

    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        raise RuntimeError(f"Artículo no encontrado en Wikipedia {WIKI_LANG}: {title}")

    return page.get("extract", "").strip(), page.get("fullurl", "")


def build_header(title: str, url: str) -> str:
    # Cabecera de procedencia (título, fuente, licencia, fecha) al inicio del .txt.
    return (
        f"# titulo: {title}\n"
        f"# fuente: {url}\n"
        f"# licencia: CC BY-SA 4.0\n"
        f"# descargado: {date.today().isoformat()}\n"
        "# ---\n"
    )


def main() -> None:
    # Descarga cada artículo faltante y lo guarda con su cabecera.
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    descargados = 0
    for source_id, title in ARTICLES.items():
        destination = RAW_DIR / f"{source_id}.txt"

        if destination.exists():
            print(f"[skip] {source_id:26s} ya existe")
            continue

        try:
            text, url = fetch_extract(title)
        except RuntimeError as exc:
            print(f"[falta] {source_id:26s} {exc}")
            continue

        if len(text) < 500:
            print(f"[aviso] '{title}' devolvió muy poco texto ({len(text)} chars)")

        destination.write_text(build_header(title, url) + text + "\n", encoding="utf-8")

        descargados += 1
        print(f"[ok] {source_id:26s} <- {title} ({len(text)} chars)")
        time.sleep(1.5)  # cortesía con la API

    total = len(list(RAW_DIR.glob("*.txt")))
    print(f"\n{descargados} artículos nuevos; {total} en total en {RAW_DIR}")


if __name__ == "__main__":
    main()
