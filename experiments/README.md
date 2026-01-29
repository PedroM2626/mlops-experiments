# ML & Deep Learning Experiments Portfolio

Este diretório contém uma série de experimentos automatizados de Machine Learning e Deep Learning, integrados com **DagsHub** e **MLflow** para rastreamento de experimentos, métricas e artefatos.

## 🚀 Experimentos Realizados

### 1. AG News Classification (Deep Learning)
- **Script**: `exp1_ag_news.py`
- **Dataset**: AG News (Notícias)
- **Modelo**: `distilbert-base-uncased` (Transformers)
- **Tarefa**: Classificação de texto em 4 categorias.
- **Métricas**: Accuracy, F1-Score, Precision, Recall.
- **Artefatos**: Modelo treinado, Tokenizer e logs de treinamento.

### 2. Temperature Forecasting (Time Series)
- **Script**: `exp2_time_series.py`
- **Dataset**: Daily Minimum Temperatures
- **Modelo**: Prophet (Facebook)
- **Tarefa**: Previsão de séries temporais de temperatura.
- **Métricas**: MAE, RMSE.
- **Artefatos**: Gráficos de forecast, componentes da série e modelo serializado (.pkl).

### 3. Twitter Sentiment Analysis (NLP)
- **Script**: `exp3_sentiment_twitter.py`
- **Dataset**: Twitter Sentiment Dataset
- **Modelo**: `distilbert-base-uncased-finetuned-sst-2-english` (Pipeline de Sentimento)
- **Tarefa**: Análise de sentimento em tweets (Zero-shot/Pre-trained).
- **Métricas**: Distribuição de sentimentos (Counts).
- **Artefatos**: Resultados em CSV e gráfico de distribuição.

## 🛠️ Como Executar

Certifique-se de que as dependências do `requirements.txt` na raiz do projeto estão instaladas e que o arquivo `.env` possui as credenciais do DagsHub.

```bash
# Para rodar a classificação de notícias
python experiments/exp1_ag_news.py

# Para rodar a previsão de temperatura
python experiments/exp2_time_series.py

# Para rodar a análise de sentimento
python experiments/exp3_sentiment_twitter.py
```

## 📊 Rastreamento (DagsHub)

Todos os experimentos são registrados automaticamente no DagsHub:
- **MLflow Tracking**: Métricas e parâmetros.
- **DagsHub Storage**: Artefatos, modelos e visualizações.

---
*Gerado automaticamente pelo MLOps Assistant.*
