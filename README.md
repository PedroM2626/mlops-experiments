# Repositório de Experimentos de Machine Learning & MLOps

Um portfólio de experimentos de ML/MLOps. Cada experimento vive na sua pasta
em `experiments/<experimento>/` com um **README acadêmico próprio** (artigo
compacto: resumo, contexto, metodologia, resultados, discussão, conclusões e
reprodução). Este documento é apenas o **índice** que conecta tudo.

Para o padrão de documentação, veja
[`docs/modelo-academico-readme.md`](docs/modelo-academico-readme.md).

---

## Índice de Experimentos

### 🧪 NLP — Análise de Sentimento, Tópicos e Representações

| Experimento | O que faz | Resultado principal | Leitura |
|---|---|---|---|
| **Grupo NLP** (senti-pred, pipelines A/B/C, Twitter Methods, Logistic multiclasse, MMoE, AG News, FE NLP) | Classificação de sentimento/tópicos e representações textuais | TF-IDF + n-grams ~0.98 F1; transformers vencem em low-data | [ver README](experiments/nlp/README.md) |
| **NLP em Regressão — Vinhos (Kaggle)** | Pontuação de vinhos por texto | Ridge MAE 1.33 / R² 0.69 vs LightGBM 1.47 / 0.63 | [ver](experiments/nlp-regression-wine/README.md) |
| **Variações Senti-Pred** | Variações do pipeline de sentimento | recorde 97.80% (TF-IDF 100k, 4-grams) | [ver](experiments/senti-pred-variations/README.md) |
| **Hierárquico 20 Newsgroups** | classificação/flat vs hieráquico, clustering | flat acc 0.7188 vs hierárquico 0.6953 | [ver](experiments/hierarchical/README.md) |

### 🤖 Reinforcement Learning / AutoML

| Experimento | Objetivo | Principal resultado | Leitura |
|---|---|---|---|
| **Q-Learning para AutoML** | agente RL otimiza hiperparâmetros | proxy RL no sales-forecast: MAE 1.4297 vs Optuna 1.4218 | [ver](experiments/reinforcement_learning/README.md) |

### 📈 Séries Temporais e Previsão

| Experimento | Objetivo | Principal resultado | Leitura |
|---|---|---|---|
| **Grupo Séries Temporais** (Prophet, benchmark 4×4, classificação 6 paradigmas, TS+NLP, forecast-classification, destilação, anomalias) | previsão, classificação e análise de TS | SARIMA vence 2/4 no benchmark; ROCKET 3/3; Logística 0.958 no forecast-direção | [ver](experiments/time_series/README.md) |
| **5 Fases de Feature Engineering (TS)** | manual vs automático vs sinais vs embeddings | DWT + manual: MAE 54.19 (melhor) | [ver](experiments/ts_fe/README.md) |
| **Sales Forecast (Hackathon)** | previsão semanal de vendas | LightGBM V2.2 MAE 1.4218 | [ver](experiments/sales-forecast/README.md) |
| **Databricks Forecast (cloud)** | Prophet/DeepAR importados (Databricks) | equivalentes locais em time_series | [ver](experiments/databricks-forecast/README.md) |

### 🖥️ Nuvem → Equivalentes Locais (Watsonx & Databricks)

| Experimento | Objetivo | Leitura |
|---|---|---|
| **IBM Watsonx (originais cloud)** | Boston Housing, Electric_Production, sentimentos | [ver](experiments/ibm-experiments/README.md) |
| **Equivalentes open-source locais** | replicar AutoML/forecast cloud (FLAML, TPOT, Prophet+Optuna, GluonTS) | [time_series](experiments/time_series/README.md) · [tabular](experiments/tabular_regression/README.md) |

### 🐱🖼️ Computer Vision

| Experimento | Objetivo | Principal resultado | Leitura |
|---|---|---|---|
| **CV Methods (CIFAR-10)** | HOG+SVM vs ResNet18 vs ViT | ViT 0.9805 vs ResNet 0.9362 vs HOG 0.3970 | [ver](experiments/computer_vision/README.md) |
| **Animal multi-label** | 4 abordagens (pets Dime/Frida) | ResNet18+aug Exact Match 1.000 | idem acima |
| **Detecção/reconhecimento facial** | LBPH, CNN, YuNet (app no notebook) | — | [ver](experiments/computer_vision/README.md) |

### 🎬 RecSys

| Experimento | Objetivo | Principal resultado | Leitura |
|---|---|---|---|
| **MovieLens RecSys — 8 abordagens** | MF, redes neurais, similaridade, heurístico | Two-Tower RMSE 0.9297; SVD mais eficiente | [ver](experiments/recommender_systems/README.md) |

### 🏎️ Regressão Tabular & AutoML local

| Experimento | Objetivo | Principal resultado | Leitura |
|---|---|---|---|
| **Grupo de Regressão Tabular** (FE tabular, Price Prediction v1–v3, IBM Watsonx local) | impacto de feature engineering; evolução de pipeline | R² 0.9489 (Random Forest); FE assimétrica por modelo | [ver](experiments/tabular_regression/README.md) |

### 🔬 Feature Selection Evolucionária

| Experimento | Objetivo | Principal resultado | Leitura |
|---|---|---|---|
| **GAAP (NSGA-II) e MO-DE vs clássicos** | seleção de features multiobjetivo (R²/F1 × nº de features) | vantagem em features interativas (California); clássicos já bastam no Twitter | [ver](experiments/feature_selection_ea/README.md) |

---

## Outros artefatos e avulsos

- **Scripts/notebooks na raiz de `experiments/`** (comparação de anomalias,
  `run_anomaly_*.py`, `run_clustering_comparison.py`, `run_senti_*.py`,
  `run_supervised_clustering.py`, `ensemble_pyramid.py`, etc.) — apoio aos
  READMEs acima.
- **Dashboard de experimentos**: `dashboard/index.html` (abrir no navegador).

## Padrões do repositório

- **Reprodução**: rodar cada script/notebook a partir da sua própria pasta;
  artefatos em `experiments/artifacts/<experimento>_<timestamp>_<sha>/`
  (`model.pkl` / `model.joblib` / `SavedModel/` / `pip_freeze.txt`); seeds
  fixas registrando no MLflow (`seed`, `git_sha`, `run_timestamp`).
- **Validação estrutural** de notebooks: `python scripts/validate_notebooks.py`
  (notebooks externos marcados como `EXT`).
- **Convenções de runtime** (CPU vs GPU) estão em cada README de grupo.

## Como navegar

1. Abra o README da pasta do experimento (links do índice acima).
2. Para os detalhes técnicos completos (código, notebooks executados),
   entre na pasta correspondente: `experiments/<experimento>/`.
3. No histórico agregado: dashboard (`dashboard/index.html`) e `mlflow ui`.

---

*Este repositório é um diário vivo de descobertas em Ciência de Dados.*