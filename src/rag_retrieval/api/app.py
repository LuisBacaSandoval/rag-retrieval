"""API de búsqueda comparativa (FastAPI).

Pendiente de implementación en el Hito 7. Expondrá, como mínimo:

- ``POST /search``: consulta + método (``bm25`` | ``dense`` | ``hybrid``) -> top-k.
- ``GET /compare?q=...``: top-k de al menos dos métodos lado a lado.
- ``GET /health`` y ``GET /metrics``.

Se adaptará el patrón de API con FastAPI del trabajo de MLOps para RAG del curso
CC0C2 (semana 14).
"""

from __future__ import annotations

# El objeto `app` de FastAPI se define en el Hito 7.
