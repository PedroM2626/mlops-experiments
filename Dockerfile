# Serving do campeão sales-forecast (FastAPI + MLflow registry local).
FROM python:3.11-slim

WORKDIR /app

# Deps de sistema p/ LightGBM/OpenCV (build enxuto, sem cache apt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Requisitos primeiro (cache de camadas). Imagem da API usa o requirements
# enxuto de MLOps (serving/monitor/retrain); o monolito fica p/ pesquisa.
COPY requirements-mlops.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements-mlops.txt

COPY . .

# Usuário não-root sem quebrar o WORKDIR /app (config espera /app/experiments/...)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

# API FastAPI (mlops.serve), não Gradio
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "mlops.serve"]
