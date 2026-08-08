# Senti-Pred — Variações do Pipeline de Análise de Sentimento

> **Área:** NLP
> **Tarefa:** Classificação de sentimentos (4 classes: Irrelevant, Negative, Neutral, Positive)
> **Métrica principal:** F1-Macro / Acurácia
> **Status:** Concluído
> **Datasets:** Twitter (Twitter Entity Sentiment — `twitter_training.csv` / `twitter_validation.csv`)

## 1. Resumo

Esta pasta consolida as **variações do projeto Senti-Pred**, unificando resultados, otimizações e lições de baselines com Transformers até ensembles de alta performance e AutoML. No dataset de tweets (4 classes), a jornada demonstrou que o refinamento dos dados e de modelos lineares robustos supera arquiteturas complexas de Deep Learning. O recorde foi alcançado por **Senti-Pred-remake2** (voting LinearSVC + LogisticRegression sobre TF-IDF 100k + 4-grams) com **97,80%** de F1-Macro/Acurácia.

## 2. Contexto e Objetivos

Explorou-se o mesmo dataset de tweets através de múltiplas abordagens — Transformers pré-treinados (RoBERTa), redes profundas, modelos lineares, ensambles, AutoML (FLAML) e engenharia de dados "Data-Centric AI" — para investigar:

- Qual a representação de texto (n-grams, vocabulário, limpeza) que maximiza o F1.
- Quanto o pré-processamento importa vs. a arquitetura do modelo.
- Como o MLOps (MLflow/DagsHub, modularidade, persistência) suporta a evolução das variações.

Hipótese central: para tweets, **pipeline de features + modelos lineares robustos superam fine-tuning de transformers** quando não há hardware massivo.

## 3. Fundamentação Teórica (curta)

- **TF-IDF / n-grams** — representação esparsa com até 4-grams e vocabulários de até 100k features p/ capturar contexto de sentiment.
- **Voting Ensemble** — combinação democrática de classificadores (LinearSVC + LogisticRegression, ou Passive Aggressive) para eliminar erros individuais.
- **Passive Aggressive** — algoritmo online que aprende rapidamente com erros, ideal para larga escala.
- **FLAML** — framework AutoML rápido (300s) para prototipagem.
- **MLflow / DagsHub** — rastreabilidade de hiperparâmetros, métricas e artefatos; wrappers persistentes (`Pipeline` + `LabelEncoder`) para inferência idêntica ao treino.

## 4. Metodologia

### 4.1 Dados
- `senti-pred-exp1/data/raw/twitter_training.csv` (treino) e `twitter_validation.csv` (treino de validação).
- 4 classes: *Irrelevant*, *Negative*, *Neutral*, *Positive*.

### 4.2 Pré-processamento (Data-Centric AI)
- **Limpeza sentiment-aware:** preservação de pontuações emocionais (`!`, `?`) e expansão de contrações.
- **Normalização de ruído:** Regex remove URLs, menções e trata caracteres repetidos (ex.: `"loooove"` → `"love"`).
- **Vetorização extrema:** n-grams até 4-grams, vocabulários de até 100k features.
- **Paralelização:** `joblib.Parallel` (15 núcleos) para lematização e limpeza em larga escala.

### 4.3 Métodos comparados
Desde modelo baseline de TF-IDF 10k + LR, passando por KNN, LinearSVC, MultinomialNB, Random Forest (Optuna), stacking (Chi2 + feature sel.), FLAML AutoML e ensambles por votação; além do Roland RoBERTa (baseline transformer). Variações isoladas em duas subpastas:

- `Senti-pred-exp1/` — pipeline completo (scripts `01_eda.py` → `04_evaluation.py`, src/api) com containerização (Dockerfile/form).
- `Senti-Pred-remake2/` — remake com `src/` modular + `data/raw/`.

### 4.4 Avaliação / MLOps
- Métrica principal: F1-Macro/Acurácia; integração **MLflow/DagsHub**.
- **Persistência:** wrappers (`Pipeline` + `LabelEncoder`) salvos via `joblib` para inferência idêntica.
- **Modularização**: cada variação isolada em diretórios p/ evitar conflitos de dependências.

### 4.5 Reprodução
- Refiro ao `EXPERIMENTS_SUMMARY.md` (resumo consolidado) e à estrutura de cada subpasta (`senti-pred-exp1/`, `Senti-Pred-remake2/`).
- Pipelines: `senti-pred-exp1/src/scripts/01_eda.py` … `04_evaluation.py`; instruções de Docker em `senti-pred-exp1/Dockerfile`/`docker-compose.yml`.
- Logs de treino versionados: `senti-pred-exp1/training_log{,_v2..v7}.txt`.

## 5. Resultados

| Modelo / Experimento | Técnica de Texto | Métrica Principal (F1-Macro/Acc) | Obs./Config |
| :--- | :--- | :--- | :--- |
| **🏆 Senti-Pred-remake2** | TF-IDF (100k) + 4-grams | **97.80%** | Record: Voting (LinearSVC + LR) |
| God Mode (Remake 1) | TF-IDF (50k) + Punct | 97.50% | Voting (Passive Aggressive + LR) |
| Ultimate (Remake 1) | TF-IDF (40k) + Char Rep | 97.00% | Correção agressiva de erros |
| FLAML (AutoML) V3 | TF-IDF (30k) + 1-2 n-grams | 96.73% | Melhor AutoML: RandomForest em 5 min |
| Insane Mode | Chi2 Feature Selection | 96.20% | Stacking Classifier (overfitting leve) |
| Logistic Regression | TF-IDF (20k) + Regex | 96.00% | Baseline linear estável |
| LinearSVC | TF-IDF (Standard) | 95.00% | Excelente para espaços esparsos |
| KNN | TF-IDF (Standard) | 95.00% | Não paramétrico, rápido |
| MultinomialNB | Trigramas + Sublinear TF | 92.06% | Busca logarítmica de alpha |
| Random Forest | Optuna (busca profunda) | 91.00% | Salto de 71% → 91% após HPO |
| Classic (LR Baseline) | TF-IDF (10k) | 87.20% | Primeiro modelo robusto (dataset total) |
| Baseline RoBERTa | Transformer (pre-trained) | ~60.00% | Lento e pouco dado (amostra de 1k) |

### Destaques por abordagem

- **AutoML (FLAML):** 96.73% em 300 segundos; selecionou `RandomForestClassifier`.
- **Ensembles por votação:** combinação LinearSVC + LogisticRegression (ou Passive Aggressive) é a mais estável.
- **Passive Aggressive:** aprende rápido com erros, ideal para larga escala (modo *Ultimate*).
- **RoBERTa:** sem hardware massivo e tiempo para fine-tuning no full, estatísticos clássicos são mais eficientes nesta tarefa.

## 6. Discussão

O comparativo mostra uma hierarquia clara: **mais n-grams + mais vocabulário + boas limpezas** elevam sistemas clássicos de 87.2% (baseline) a **97.8%** (record), enquanto o RoBERTa ficou em ~60% por falta de dados/hardware. A votação de modelos lineares robustos (SVC + LR) foi o fator-sorda para o recorde. As escolhas *Data-Centric* (retoção de URLs, n-grams de caracteres e vocabulário de 100k) superaram a modelagem de arquiteturas complexas. Limitações: FLAMB escolhe RandomForest, mas os ensembles lineares venceram com mais features; o overfitting leve foi reportado no *Insane Mode* (stacking com Chi2).

## 7. Conclusões e Recomendações

- **Priorizar features esparsas ricas (TF-IDF, até 4-grams, 100k) + voting de modelos lineares** como melhor custo vs. com para este domínio de tweets.
- Para **prototipagem rápida**, `AutoML (FLAML)` é suficiente em 5 min (96.73%).
- **Transformers** só valem com hardware e dataset completo (ver `../nlp/README.md` para fine-tuning).
- **Próximos passos:** interface Streamlit para comparar modelos em tempo real; deploy via Docker para reprodutibilidade; testar LLMs zero-shot (API/quantizados).

## 8. Referências e Arquivos

- `EXPERIMENTS_SUMMARY.md` — resumo consolidado (fonte desta documentação).
- `senti-pred-exp1/` — pipeline original (scripts 01–04, Docker, MLflow local, logs de treino).
- `Senti-Pred-remake2/` — remake modular com `src/` e dados raw.
- Casos semelhantes (representação/ensembles, logistic multi): `../nlp/README.md`.