# Serving for the sales-forecast champion (FastAPI + local MLflow registry).
FROM python:3.11-slim

WORKDIR /app

# System deps for LightGBM/OpenCV (lean build, no apt cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Requirements first (layer caching). The API image uses the lean MLOps
# requirements (serving/monitor/retrain); the monolith stays for research.
COPY requirements-mlops.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements-mlops.txt

COPY . .

# Non-root user without breaking WORKDIR /app (config expects /app/experiments/...)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

# FastAPI API (mlops.serve), not Gradio
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "mlops.serve"]
