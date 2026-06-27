FROM python:3.10-slim

WORKDIR /app

# Instala bibliotecas do sistema necessárias para compilar dependências (como lightgbm)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia os requirements e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código para o container
COPY . .

# Comando padrão
CMD ["python", "scripts/train.py", "--n_trials", "20"]
