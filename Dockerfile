# Use uma imagem base oficial do Python 3.11
FROM python:3.11-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema necessárias para bibliotecas de ML (Auto-sklearn, XGBoost, etc)
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar os arquivos de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar as dependências do Python
# Usamos --no-cache-dir para reduzir o tamanho da imagem
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código do projeto
COPY . .

# Criar um usuário não-root para segurança (recomendado pelo Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Definir o diretório de trabalho para o diretório do usuário
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expor a porta que o Gradio usa
EXPOSE 7860

# Comando para rodar a aplicação
# O Gradio no Spaces espera rodar na porta 7860
CMD ["python", "app.py"]
