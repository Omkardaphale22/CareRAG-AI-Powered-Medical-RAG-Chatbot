FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw_documents data/vector_store data/processed

# Build sample vector store (used if ./data volume is empty on first run)
RUN python scripts/ingest_data.py --source sample

# 8000 = FastAPI (JSON), 8501 = Streamlit UI
EXPOSE 8000 8501

# Default command = API. docker-compose overrides it for the Streamlit service.
CMD ["python", "-m", "src.api"]
