"""Construye el corpus troceado a partir del texto fuente.

Lee los artículos de ``data/corpus/raw/*.txt`` (descargados con
``fetch_sources.py``), les quita la cabecera de procedencia, limpia el texto y lo
divide en fragmentos solapados. El resultado se guarda en
``data/corpus/corpus.jsonl``, una línea por fragmento::

    {"doc_id": "bm25__c03", "source_id": "bm25", "title": "Okapi BM25", "text": "..."}

``doc_id`` es la unidad que indexan y devuelven los recuperadores; ``source_id``
identifica el artículo de origen y se usa para mapear los juicios de relevancia
(que se etiquetan a nivel de artículo). El troceado por caracteres con solape
sigue el patrón de índice del trabajo de MLOps para RAG del curso CC0C2
(semana 14). Solo usa la biblioteca estándar, así que corre igual en local y en
Docker; el orden de salida es determinista para que el corpus sea reproducible.

Uso:
    python scripts/prepare_corpus.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "corpus" / "raw"
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus" / "corpus.jsonl"

CHUNK_SIZE = 800  # caracteres por fragmento
OVERLAP = 120  # caracteres compartidos entre fragmentos consecutivos
MIN_CHUNK_CHARS = 200  # descarta colas demasiado cortas para ser útiles

# Secciones finales de un artículo de Wikipedia que no son prosa útil para
# recuperar. El cuerpo se corta en la primera que aparezca.
TERMINAL_SECTIONS = {
    "véase también",
    "referencias",
    "notas y referencias",
    "notas",
    "bibliografía",
    "enlaces externos",
}


def parse_raw(path: Path) -> tuple[str, str]:
    """Devuelve (título, cuerpo) separando la cabecera de procedencia."""
    lines = path.read_text(encoding="utf-8").splitlines()

    title = path.stem
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("# titulo:"):
            title = line.split(":", 1)[1].strip()
        if line.strip() == "# ---":
            body_start = i + 1
            break

    return title, "\n".join(lines[body_start:])


def strip_tail_sections(text: str) -> str:
    """Corta el cuerpo en la primera sección final no-prosa (Referencias, etc.)."""
    lineas = text.splitlines()
    for i, linea in enumerate(lineas):
        if linea.strip().lower() in TERMINAL_SECTIONS:
            return "\n".join(lineas[:i])
    return text


def clean_text(text: str) -> str:
    """Normaliza espacios y descarta líneas triviales (encabezados sueltos)."""
    lineas_utiles = []
    for linea in strip_tail_sections(text).splitlines():
        linea = linea.strip()
        if len(linea) < 3:  # líneas vacías o encabezados de sección de una palabra
            continue
        lineas_utiles.append(linea)

    unido = " ".join(lineas_utiles)
    return re.sub(r"\s+", " ", unido).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Trocea el texto en ventanas de ``chunk_size`` con ``overlap`` de solape."""
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    # Fusiona una cola muy corta con el fragmento anterior para no dejar restos.
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        cola = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {cola}".strip()

    return chunks


def build_corpus() -> list[dict[str, str]]:
    if not RAW_DIR.exists():
        raise RuntimeError(f"No existe {RAW_DIR}. Ejecuta antes scripts/fetch_sources.py")

    registros: list[dict[str, str]] = []

    for path in sorted(RAW_DIR.glob("*.txt")):
        source_id = path.stem
        title, body = parse_raw(path)
        texto = clean_text(body)

        for i, chunk in enumerate(chunk_text(texto)):
            registros.append(
                {
                    "doc_id": f"{source_id}__c{i:02d}",
                    "source_id": source_id,
                    "title": title,
                    "text": chunk,
                }
            )

    if not registros:
        raise RuntimeError("El corpus quedó vacío; revisa los archivos de raw/")

    return registros


def main() -> None:
    registros = build_corpus()

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8") as file:
        for registro in registros:
            file.write(json.dumps(registro, ensure_ascii=False) + "\n")

    fuentes = sorted({r["source_id"] for r in registros})
    longitudes = [len(r["text"]) for r in registros]
    media = sum(longitudes) / len(longitudes)

    print(f"Fuentes: {len(fuentes)}")
    print(f"Fragmentos: {len(registros)}")
    print(f"Longitud media por fragmento: {media:.0f} caracteres")
    print(f"Guardado en {CORPUS_PATH}")


if __name__ == "__main__":
    main()
