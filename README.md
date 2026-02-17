# Repositório de Experimentos de Machine Learning & MLOps

Este repositório é dedicado a registrar a jornada de aprendizado, experimentos práticos e a evolução de modelos de Machine Learning, com foco em NLP e MLOps. O objetivo principal é documentar como diferentes abordagens, arquiteturas e engenharia de features impactam os resultados reais.

## 🧪 Experimentos de NLP: Análise de Sentimento (Senti-Pred)

Realizei uma série de experimentos comparando modelos manuais e frameworks de AutoML em dois cenários distintos de pré-processamento para classificação de sentimentos em reviews de redes sociais.

### 🧠 Principais Aprendizados e Descobertas (NLP)

#### 1. A "Relatividade" dos Modelos (No Free Lunch)
A maior lição destes experimentos foi que **não existe um "modelo perfeito" universal**. A performance de um algoritmo é totalmente dependente do contexto dos dados e das decisões de pré-processamento.
- O **LinearSVC** variou de **0.74** (F1-Macro) em um cenário para **0.94** em outro, simplesmente por ajustes no vocabulário e n-grams (mesmo com o mesmo dataset e pré-processamento).
- Modelos simples como **KNN** superaram frameworks complexos de AutoML em casos específicos, provando que a complexidade nem sempre é sinônimo de superioridade.

#### 2. O Poder da Engenharia de Features (TF-IDF + N-grams)
A diferença entre um modelo medíocre e um estado-da-arte muitas vezes reside na forma como o texto é transformado em números:
- **Unigramas vs Bigramas**: A inclusão de bigramas (`ngram_range=(1,2)`) foi crucial para capturar contextos como "não é bom", permitindo que modelos lineares entendessem a negação.
- **Tamanho do Vocabulário**: Limitar excessivamente as features (`max_features`) pode cegar o modelo, enquanto um vocabulário muito vasto pode introduzir ruído. O equilíbrio em **15.000 features** mostrou-se ideal para este dataset.

#### 3. Deep Learning vs Modelos Clássicos
- **Fine-tuning de Transformers**: No experimento [exp1_ag_news.py](experiments/exp1_ag_news.py), utilizei o **DistilBERT** para classificação de notícias, atingindo alta acurácia rapidamente através de Transfer Learning. Isso mostra que, para tarefas complexas de semântica, modelos pré-treinados superam o TF-IDF manual.

---

## 📈 Séries Temporais e Previsão (Forecast)

Explorei diferentes abordagens para predição de dados temporais, desde modelos estatísticos clássicos até algoritmos de Gradient Boosting otimizados.

### 🧠 Principais Aprendizados e Descobertas (Time Series)

#### 1. Prophet vs Gradient Boosting
- No experimento [exp2_time_series.py](experiments/exp2_time_series.py), utilizei o **Prophet** (Meta) para prever temperaturas diárias. O Prophet é excelente para capturar sazonalidades (diária, semanal, anual) de forma automática e robusta a feriados.
- Já no projeto [sales forecast](experiments/sales%20forecast), o foco foi no **LightGBM** com **Optuna**. Aprendi que para séries temporais com muitas features externas, o Gradient Boosting com lags manuais e janelas móveis tende a ser mais preciso que modelos puramente estatísticos.

#### 2. Engenharia de Features Temporais
A "inteligência" do modelo de vendas veio da criação de features que capturam o tempo:
- **Lags e Rolling Windows**: Ensinar ao modelo o que aconteceu há 1, 4 e 52 semanas foi vital para capturar sazonalidades anuais.
- **Features Cíclicas**: Transformar semanas em coordenadas de seno/cosseno permitiu ao modelo entender que a semana 52 está próxima da semana 1.

---

## 🤖 AutoML e MLOps Studio

*(Esta seção será expandida conforme o desenvolvimento do [AutoMLOps-Studio](experiments/AutoMLOps-Studio) avança.)*

---

## 🛠️ Estrutura de Experimentos

### Projetos Analisados:
1. **[senti-pred](experiments/senti-pred)**: Foco em AutoML e exploração de múltiplos frameworks.
2. **[old_senti-pred_upgrade](experiments/old_senti-pred_upgrade)**: Foco em modelos manuais clássicos (LinearSVC, KNN, RF, MLP) e otimização de pipeline TF-IDF.
3. **[sales forecast](experiments/sales%20forecast)**: Foco em Séries Temporais, LightGBM e Otimização Bayesiana.
4. **[ibm-experiments](experiments/ibm-experiments)**: Notebooks exploratórios de Boston Housing e produções elétricas usando Snap ML da IBM.
5. **[databricks forecast](experiments/databricks%20forecast)**: Script de download de artefatos para integração com Databricks.

### Experimentos Rápidos:
- **[exp1_ag_news.py](experiments/exp1_ag_news.py)**: Classificação de notícias com DistilBERT.
- **[exp2_time_series.py](experiments/exp2_time_series.py)**: Previsão de temperatura com Prophet.
- **[exp3_sentiment_twitter.py](experiments/exp3_sentiment_twitter.py)**: Análise de sentimento zero-shot com HuggingFace Pipelines.

---

## 🚀 Conclusão Final: A Performance é Sistêmica

Após dezenas de experimentos, a maior lição é que a métrica final não é um mérito exclusivo do algoritmo escolhido.
- **Simbiose**: Quase qualquer modelo pode atingir métricas de excelência se o pipeline (pré-processamento e hiperparâmetros) for exaustivamente otimizado para ele.
- **Eficiência vs. Força Bruta**: O desafio do Cientista de Dados não é apenas "chegar no 0.99", mas sim encontrar o modelo que chega lá de forma natural e eficiente, sem "lutar contra a natureza dos dados".
- **Decisão Estratégica**: Escolher o modelo certo é, na verdade, escolher o caminho de menor resistência entre os dados brutos e a predição precisa.

---
*Este repositório é um diário vivo de descobertas em Ciência de Dados.*
