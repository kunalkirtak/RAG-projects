# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss / sentence-transformers / pdf parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/documents data/vectorstore logs

EXPOSE 8000 8501

# Default: run the FastAPI service. Override CMD to run Streamlit instead:
#   docker run <image> streamlit run streamlit_app.py --server.address=0.0.0.0
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
