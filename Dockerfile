FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

# El índice se construye en el Hito 4; hasta entonces esta línea queda comentada.
# RUN python scripts/build_index.py

EXPOSE 8000

# El objeto `app` de FastAPI se define en el Hito 7.
CMD ["uvicorn", "rag_retrieval.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
