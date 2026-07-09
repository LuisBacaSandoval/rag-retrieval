FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cachea el modelo de embeddings en la imagen (reproducibilidad offline).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

ENV PYTHONPATH=/app/src

# data/indexes está en .dockerignore, así que el índice se construye aquí.
RUN python scripts/build_index.py

EXPOSE 8000

CMD ["uvicorn", "rag_retrieval.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
