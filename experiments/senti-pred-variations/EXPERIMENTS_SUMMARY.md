# Resumo Consolidado dos Experimentos: Análise de Sentimento (Senti-Pred)

Este documento unifica os resultados, otimizações e aprendizados obtidos em todas as variações do projeto Senti-Pred, desde baselines com Transformers até Ensembles de alta performance e AutoML.

## 1. Contexto e Evolução
Exploramos o dataset do Twitter (4 classes: *Irrelevant*, *Negative*, *Neutral*, *Positive*) através de múltiplas abordagens. A jornada demonstrou que, para este dataset, o refinamento dos dados e modelos lineares robustos superam arquiteturas complexas de Deep Learning.

## 2. Comparativo de Performance (Resultados Consolidados)

| Modelo / Experimento | Técnica de Texto | Métrica Principal (F1-Macro / Acc) | Observações |
| :--- | :--- | :--- | :--- |
| **🏆 Senti-Pred-remake2** | **TF-IDF (100k) + 4-grams** | **97.80%** | **Recorde**: Voting (LinearSVC + LR). |
| **God Mode (Remake 1)** | TF-IDF (50k) + Punct | 97.50% | Voting (Passive Aggressive + LR). |
| **Ultimate (Remake 1)** | TF-IDF (40k) + Char Rep | 97.00% | Foco em correção agressiva de erros. |
| **FLAML (AutoML) V3** | TF-IDF (30k) + 1-2 n-grams | 96.73% | Melhor AutoML: RandomForest em 5 min. |
| **Insane Mode** | Chi2 Feature Selection | 96.20% | Stacking Classifier (Overfitting leve). |
| **Logistic Regression** | TF-IDF (20k) + Regex | 96.00% | Baseline linear extremamente estável. |
| **LinearSVC** | TF-IDF (Standard) | 95.00% | Excelente para espaços esparsos. |
| **KNN** | TF-IDF (Standard) | 95.00% | Abordagem não paramétrica rápida. |
| **MultinomialNB** | Trigramas + Sublinear TF | 92.06% | Otimizado via busca logarítmica de Alpha. |
| **Random Forest** | Optuna (Busca Profunda) | 91.00% | Salto de 71% -> 91% após HPO. |
| **Classic (LR Baseline)** | TF-IDF (10k) | 87.20% | Primeiro modelo robusto com dataset total. |
| **Baseline RoBERTa** | Transformer (Pre-trained) | ~60.00% | Lento e pouco dado (amostra de 1k). |

## 3. Otimizações de Engenharia de Dados
O maior diferencial de performance veio do pré-processamento ("Data-Centric AI"):

- **Limpeza Sentiment-Aware**: Preservação de pontuações emocionais (!, ?) e expansão de contrações.
- **Normalização de Ruído**: Uso de Regex para remover URLs, menções e tratar caracteres repetidos (ex: "loooove" -> "love").
- **Vetorização Extrema**: Uso de n-grams (até 4-grams) e vocabulários de até 100k features para capturar nuances contextuais.
- **Paralelização**: Uso de `joblib.Parallel` (15 núcleos) para lematização e limpeza em larga escala.

## 4. Destaques por Abordagem

### 🤖 AutoML (FLAML)
O framework FLAML provou ser a melhor ferramenta para prototipagem rápida, atingindo **96.73%** em apenas 300 segundos, selecionando automaticamente o `RandomForestClassifier` como vencedor.

### 🏛️ Modelos Lineares e Ensembles
- **Voting Ensemble**: A combinação de `LinearSVC` e `LogisticRegression` (ou Passive Aggressive) provou ser a mais estável, eliminando erros individuais através de "votação democrática".
- **Passive Aggressive**: Utilizado no modo *Ultimate* por sua capacidade de aprender rapidamente com erros de classificação, sendo ideal para datasets de larga escala.

### 🧠 Deep Learning vs. Classic
A tentativa inicial com **RoBERTa** mostrou que, sem hardware massivo e tempo para fine-tuning no dataset completo, modelos estatísticos clássicos são mais eficientes e precisos para este domínio específico de tweets.

## 5. Arquitetura e MLOps
- **Modularização**: Cada variação isolada em diretórios para evitar conflitos de dependências.
- **Rastreabilidade**: Integração com **MLflow** e **DagsHub** para log de hiperparâmetros, métricas e artefatos.
- **Persistência**: Uso de wrappers customizados (`Pipeline` + `LabelEncoder`) salvos via `joblib` para garantir inferência idêntica ao treino.

## 6. Próximos Passos
- Implementar interface unificada em Streamlit para comparar os modelos em tempo real.
- Realizar deploy via Docker para garantir reprodutibilidade em qualquer ambiente.
- Explorar LLMs (via API ou quantizados) para análise de sentimento de nível zero-shot.
