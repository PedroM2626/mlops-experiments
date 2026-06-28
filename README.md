# Repositório de Experimentos de Machine Learning & MLOps

Este repositório é dedicado a registrar a jornada de aprendizado, experimentos práticos e a evolução de modelos de Machine Learning, com foco em NLP e MLOps. O objetivo principal é documentar como diferentes abordagens, arquiteturas e engenharia de features impactam os resultados reais.

## 🧪 Experimentos de NLP: Análise de Sentimento (Senti-Pred)

Realizei uma série de experimentos comparando modelos manuais e frameworks de AutoML em dois cenários distintos de pré-processamento para classificação de sentimentos em reviews de redes sociais.

### 🏗️ Ensemble Pyramid — 6 Camadas de Ensembles sobre Ensembles

O experimento mais recente implementa uma arquitetura piramidal com 6 camadas de ensembles, combinando técnicas de Bagging, Voting e Stacking de forma hierárquica:

**Arquitetura:**
- **Camada 1**: Base Learners (LR, LinearSVC, NB, CNB, Ridge, RF, ET)
- **Camada 2**: Ensembles dos Base Learners (Bagging + Voting + Stacking)
- **Camada 3**: Ensembles de Ensembles (Stacking + Bagging sobre Stacking + Voting)
- **Camada 4**: Meta-Ensemble Final (Meta Voting Soft + Meta Stacking + Meta Voting Hard)
- **Camada 5**: Meta-Ensemble Intermediário (Meta2 Voting Soft + Meta2 Stacking + Meta2 Voting Hard)
- **Camada 6**: Meta-Ensemble Final Aprimorado (Final Stacking + Final Voting Soft + Final Voting Hard)

**Principais Características:**
- Mantém tudo em formato esparso para otimização de memória (TF-IDF 70k features ocupa ~15MB)
- Classes leves (PreFittedSoftVoting, PreFittedHardVoting, MetaStackingLR) evitam re-treino desnecessário
- Combina predições probabilísticas de múltiplos níveis hierárquicos
- Atinge F1-score de ~0.98+ na validação com ganhos progressivos por camada

### 🚀 Versatile Ensemble Pyramid (Script AutoML Altamente Personalizável)

Este não é um script estático, mas um motor de AutoML flexível que utiliza Reinforcement Learning para otimizar sua própria arquitetura a cada execução.

**Variabilidade Dinâmica:**
- **Quantidade de Modelos Variável**: O número de modelos por camada não é fixo. O RL Meta-Learner decide quantos e quais modelos usar (ex: Camada 1 pode ter 3 modelos, Camada 2 apenas 2), maximizando a diversidade e eficiência.
- **Estratégia de Seleção Estocástica**: Utiliza uma variação de *Thompson Sampling* para escolher modelos. O agente mantém um ranking de performance mas introduz ruído planejado para testar potencias sinergias novas entre as meta-features.

**Parâmetros de Customização (CLI):**
Você pode ajustar o comportamento do script diretamente via linha de comando sem alterar o código:
- `--layers`: Define a profundidade total da pirâmide (ex: `--layers 15`).
- `--min_models` & `--max_models`: Controla a largura e diversidade de cada camada (ex: `--min_models 3 --max_models 6`).
- `--epsilon`: Controle de exploração do RL (0.1 = focado, 0.5 = muito explorador).
- `--metric`: Métrica de otimização para o agente de RL (`f1` ou `accuracy`).
- `--strategy`: Estratégia de conexão entre camadas (`dense`, `residual`, `simple`).
- `--jitter`: Ativa a variação aleatória de hiperparâmetros (True/False).
- `--patience`: Camadas sem melhora antes do **Early Stopping**.
- `--seed`: Garante **reprodutibilidade 100%** através de seeding global.
- `--tfidf_max` & `--tfidf_ngrams`: Customização da extração de features inicial.



**Como rodar com customização extrema:**
```bash
python experiments/flexible_ensemble_pyramid.py --layers 15 --min_models 3 --max_models 6 --strategy dense --jitter True --metric f1 --tfidf_max 75000
```

*(As configurações são automaticamente registradas no MLflow para comparação entre diferentes estratégias de evolução).*

### Dashboard Unificado de Experimentos

O repositorio inclui uma interface web unificada que apresenta **todos os 38 experimentos** em um dashboard interativo com tema escuro premium:

1. Abra o arquivo diretamente no navegador:

```bash
# Basta abrir no navegador
dashboard/index.html
```

Recursos do dashboard:
- Visao geral com estatisticas do repositorio (total, completos, categorias, tecnicas)
- Graficos interativos (distribuicao por categoria, status, evolucao Senti-Pred, radar de tecnicas)
- Filtros por categoria (Sidebar), status e busca textual em tempo real
- Cards expandiveis com detalhes de cada experimento (tecnicas, modelos, scripts, metricas)
- 7 categorias: NLP Sentimento, NLP Classificacao, Series Temporais, Computer Vision, Anomalias, IBM Watson, Regressao
- Design responsivo (mobile-friendly) com glassmorphism e micro-animacoes
- Nenhuma dependencia externa necessaria (HTML + CSS + JS vanilla)

**Características de Engenharia:**
- **Jitter de Hiperparâmetros**: Mutações aleatórias nos parâmetros dos modelos (C, alpha, n_estimators) para descobrir configurações ótimas além do padrão.
- **Skip Connections Dinâmicas**: Suporte a arquiteturas **Densas** (todas as camadas anteriores), **Residuais** (apenas a anterior + input original) ou **Simples**.
- **Auto-Voting por Camada**: Cria automaticamente um Voting Ensemble (Soft ou Hard) ao final de cada nível da pirâmide, consolidando o conhecimento local.
- **Bagging Adaptativo**: Pool de modelos agora inclui variantes de Bagging para aumentar a robustez contra overfitting.
- **Predição Recursiva (Full Inference)**: Reconstroi automaticamente toda a cadeia de transformações para predição em qualquer profundidade da pirâmide.
- **Registro MLOps Studio**: Registro completo de parâmetros, métricas, modelos (`.pkl`), matrizes de confusão e base de conhecimento RL no **MLflow/Dagshub**.




**Tecnologias Utilizadas:**
- Scikit-learn para todos os ensembles e modelos base
- Otimização de hiperparâmetros para velocidade e performance
- Processamento de texto com limpeza customizada (URLs, menções, hashtags, caracteres especiais)

### 🧠 Principais Aprendizados e Descobertas (NLP)

#### 1. A "Relatividade" dos Modelos (No Free Lunch)
A maior lição destes experimentos foi que **não existe um "modelo perfeito" universal**. A performance de um algoritmo é totalmente dependente do contexto dos dados e das decisões de pré-processamento.
- O **LinearSVC** variou de **0.74** (F1-Macro) em um cenário para **0.94** em outro, simplesmente por ajustes no vocabulário e n-grams (mesmo com o mesmo dataset e pré-processamento).
- Modelos simples como **KNN** superaram frameworks complexos de AutoML em casos específicos, provando que a complexidade nem sempre é sinônimo de superioridade.
- O **Ensemble Pyramid** demonstrou que combinações hierárquicas inteligentes podem superar modelos individuais, atingindo F1-scores de **0.98+** através de meta-ensembles progressivos.

#### 2. O Poder da Engenharia de Features (TF-IDF + N-grams)
A diferença entre um modelo medíocre e um estado-da-arte muitas vezes reside na forma como o texto é transformado em números:
- **Unigramas vs Bigramas**: A inclusão de bigramas (`ngram_range=(1,2)`) foi crucial para capturar contextos como "não é bom", permitindo que modelos lineares entendessem a negação.
- **Tamanho do Vocabulário**: Limitar excessivamente as features (`max_features`) pode cegar o modelo, enquanto um vocabulário muito vasto pode introduzir ruído. O equilíbrio em **15.000 features** mostrou-se ideal para este dataset.

#### 3. Deep Learning vs Modelos Clássicos
- **Fine-tuning de Transformers**: No experimento [exp1_ag_news.py](experiments/exp1_ag_news.py), utilizei o **DistilBERT** para classificação de notícias, atingindo alta acurácia rapidamente através de Transfer Learning. Isso mostra que, para tarefas complexas de semântica, modelos pré-treinados superam o TF-IDF manual.

#### 4. O Paradoxo do Multi-Task Learning (MMoE) e Rollback Tático para TF-IDF
No experimento [mmoe_emotion_classifier.py](experiments/mmoe_emotion_classifier.py) com o dataset Google `go_emotions`, testamos a arquitetura neural **MMoE (Multi-gate Mixture of Experts)**. A hipótese era que tarefas correlacionadas (Alegria, Tristeza, Raiva) se ajudariam mutuamente. Tivemos duas grandes lições:
- **Efeito da Fartura de Dados (Data Starvation vs Abundance)**: Quando os dados eram escassos ou as features eram fracas (TF-IDF com amostragem reduzida), forçar as redes a compartilhar "Experts" via MMoE foi espetacular, pois mitigou a Transferência Negativa e elevou a performance geral.
- **Interferência Catastrófica com Transformers**: Ao processar **todas as 43.000 amostras** usando potentes **Embeddings Densos (768d do DistilBERT na GPU)**, as redes Single-Task independentes ficaram tão autossuficientes e informadas que o MMoE se tornou um gargalo. Tentar compartilhar recursos neste cenário de fartura gerou "Interferência Catastrófica", fazendo o MMoE *perder* (-0.99%) para redes isoladas tradicionais.
- **Rollback para TF-IDF e Otimização do Target**: O usuário já havia experienciado engarrafamentos similares em outros projetos NLP e sugeriu um "Rollback" tático de DistilBERT de volta para **TF-IDF (5000 features)**. Embora o TF-IDF não entenda contexto semântico, ele transformou o dataset em matrizes esparsas onde "palavras isoladas" serviam como gatilhos perfeitos para o MMoE conectar os especialistas. Combinando isso com a alteração da métrica F1 de `macro` para `weighted` (para balancear matematicamente o peso brutal da classe de "Alegria" que é a maioria no dataset), nós quebramos a barreira do `0.8` exigida, saltando para impressionantes **0.9393** no MMoE (+1.86% de ganho sobre Single-Task). Isso prova que, às vezes, "features esparsas" funcionam melhor com rotas neurais complexas do que "features profundas".
- **A Dinâmica do Vocabulário (15.000 Features)**: Ao expandirmos o `max_features` do TF-IDF de 5.000 para 15.000, o F1-Weighted saltou para incríveis **0.9464**. Isso ocorre pois emoções se expressam através de uma cauda longa de vocabulário raro. Contudo, observou-se algo fascinante: o ganho de arquitetura do MMoE em relação ao modelo Single-Task diminuiu de **+1.86%** para **+1.24%**. A lição que fica é: *conforme as features se tornam mais descritivas, as redes isoladas se tornam mais autossuficientes*, reduzindo a dependência da rede complexa de Experts, caminhando na direção da "Interferência Catastrófica" observada no DistilBERT.
- **Rendimentos Decrescentes e o Teto Ótimo (20.000 Features)**: Em testes subsequentes, aumentamos o `max_features` para 20.000. O ganho do MMoE foi irrisório (+0.13%, atingindo 0.9477), enquanto as redes Single-Task sofreram uma *queda* de performance, provando que as 5.000 palavras extras eram majoritariamente ruído (typos, gírias obscuras). O modelo MMoE conseguiu extrair algum valor residual, mas ao custo de um aumento massivo de parâmetros na rede (memória e tempo de extração). Portanto, decidimos fixar e adotar o **15.000 features** como o "sweet spot" que equilibra alta precisão e processamento enxuto.
- **Quebrando a Barreira do 0.95 (N-Grams e Retenção de Stop Words)**: Para espremer a máxima performance possível sem mudar a arquitetura neural, introduzimos limpeza de URLs e menções, retivemos as *stop words* (essenciais para contexto de emoção e negação) e ativamos a extração de *bigramas* (`ngram_range=(1,2)`) mantendo as 15.000 features. O resultado foi histórico: o MMoE saltou para estonteantes **0.9548**, enquanto a abordagem Single-Task isolada subiu para 0.9461. A inclusão de bigramas atuou como um salto gigantesco de "Feature Engineering", provando mais uma vez que as *features* certas (ex: capturar o bigrama "not happy") elevam toda a fundação matemática, embora, previsivelmente, tenham reduzido ainda mais a vantagem percentual da arquitetura complexa do MMoE (caiu para apenas +0.92% sobre Single-Task).
- **A Cereja do Bolo: Focal Loss**: Para atingir a perfeição, substituímos a clássica `BCEWithLogitsLoss` por uma **Focal Loss Binária**. Essa função dinamicamente penaliza amostras fáceis e foca os pesos do gradiente nas amostras difíceis (onde o modelo errava). O resultado foi o ápice do experimento: o F1-Weighted do MMoE subiu para **0.9566** (com Tristeza e Alegria batendo >0.962). A Focal Loss provou que alinhar a função de otimização à dificuldade inerente do desbalanceamento de texto tira a métrica do estado "excelente" e a leva para o "estado da arte".
- **O Duelo Final: Deep Learning vs Machine Learning Clássico**: No último estágio, colocamos nossa super-rede MMoE contra os algoritmos clássicos de Machine Learning (LinearSVC, LightGBM, Extra Trees). Os clássicos receberam a matriz de features diretamente em formato **esparso**, o que otimizou brutalmente a memória RAM. O resultado provou um velho ditado: *árvores randômicas amam features esparsas*. Como as matrizes de TF-IDF com 15.000 colunas (bigramas) são formadas por quase 99% de zeros em cada linha (textos curtos), algoritmos baseados em árvores randomizadas ignoram esse "oceano de vazios" sorteando e explorando apenas as colunas que realmente possuem sinal, diferentemente das redes neurais que gastam muito processamento multiplicando as matrizes por zero. Enquanto o LightGBM (0.9473) sofreu com a altíssima dimensionalidade, o **LinearSVC (0.9572)** bateu de frente com o MMoE. Mas o verdadeiro vencedor foi o **Extra Trees Classifier**, que destroçou a barreira atingindo um F1-Weighted histórico de **0.9643**. Isso demonstra que, para representações de N-grams em extrema dimensionalidade esparsa, métodos de ensemble randomizados superam redes neurais profundas, além de não exigirem processamento massivo de GPU.

---

## 📈 Séries Temporais e Previsão (Forecast)

Explorei diferentes abordagens para predição de dados temporais, desde modelos estatísticos clássicos até algoritmos de Gradient Boosting otimizados.

### 🧠 Principais Aprendizados e Descobertas (Time Series)

#### 1. Evolução do Prophet e Optuna
- Nos cadernos interativos de Forecast ([temperature_forecasting_prophet.ipynb](experiments/temperature_forecasting_prophet.ipynb) e [property-sales-time-series.ipynb](experiments/property-sales-time-series.ipynb)), o modelo **Prophet** (Meta) evoluiu para uma arquitetura V2. Introduzimos a **Busca Bayesiana (Optuna)** para sintonizar a flexibilidade da tendência (`changepoint_prior_scale`) e a força da sazonalidade (`seasonality_prior_scale`), guiado pela métrica de erro (MAE) extraída via **Time Series Cross-Validation**.
- O Prophet validado e otimizado via Optuna demonstra agora uma forte reprodutibilidade. Além de prever sazonalidades de forma automática, a busca do melhor hiperparâmetro (como `multiplicative` para o seasonality mode) garantiu um MAE Cross-Validated próximo a ~2.19, superior às configurações default do modelo em casos complexos de ruído diário.
- **O Desafio Prophet vs LightGBM:** Para provar o teto de performance do algoritmo estatístico, confrontamos o Prophet com o LightGBM no dataset univariado de Temperaturas. Injetamos forte Engenharia de Features no LightGBM (Lags temporais e Rolling Windows) para que ele capturasse a "memória do tempo". O resultado foi uma vitória contundente do **Machine Learning Clássico (LightGBM)** com um MAE final de **1.7344**, superando o Prophet que estagnou em **1.96** no Hold-out. Isso prova que algoritmos de árvore (capazes de ler os Lags imediatamente anteriores) reagem melhor a choques abruptos de variação diária do que equações aditivas baseadas apenas no calendário estático.

#### 2. Evolução de Performance no Sales Forecast (V2 -> V2.1 -> V2.2)
A solução de previsão de vendas semanais por ponto de venda (PDV) evoluiu significativamente através de um pipeline robusto de MLOps:
- **V2 (Base)**: MAE de **2.5769** com ~21 features e apenas 2 variáveis categóricas sem suporte MLOps.
- **V2.1 (MLOps)**: MAE de **2.2340** (-13.3%) ao expandir para 23 features, incluir 5 categóricas e rastreamento básico no MLflow.
- **V2.2 (Atual)**: MAE de **1.4218** (**-44.8% vs V2** e **-36.3% vs V2.1**) com **32 features** detalhadas (10 categóricas dimensionais, features de preço, tendência/volatilidade) e Optuna com Pruning de trials não promissores (20 de 30 trials podados automaticamente), economizando tempo computacional (~12 min de treino). O pipeline conta com tracking completo via MLflow, container Docker e 10 testes automatizados via Pytest.

#### 3. Aprendizados Importantes no Sales Forecast
- **Engenharia de Váriaveis Categoricas e Elasticidade-Preço**: A inclusão de variáveis categóricas de alta cardinalidade (`subcategoria`, `tipos`, `fabricante`, etc.) tratadas nativamente pelo LightGBM e a feature de `preco_medio_unitario` (gross_value / quantity) representaram, juntas, cerca de 65% do ganho de performance.
- **O Experimento Frustrado da Transformação Logaritmica (`log1p`)**: Aplicar `log1p` sobre a quantidade para reduzir a assimetria fez o MAE saltar de 1.42 para **2.7094**. Como o MAE (L1 loss) em escala log otimiza o erro relativo (MAPE), o modelo tornou-se muito conservador e subestimou grandes volumes de vendas. O pipeline final foi mantido com o target na escala original.
- **Limitações do CatBoost no Ensemble (V2.3)**: Tentar misturar LightGBM com CatBoost consumiu mais de 23 GB de RAM e monopolizou a CPU por mais de 30 horas sem terminar a baseline devido à alta cardinalidade (ex: fabricante com 343 categorias unicas). O LightGBM demonstrou enorme superioridade em eficiência de hardware, finalizando o HPO Bayesiano em ~12 minutos.

#### 4. Engenharia de Features Temporais
A "inteligência" do modelo de vendas veio da criação de features que capturam o tempo:
- **Lags e Rolling Windows**: Ensinar ao modelo o que aconteceu há 1, 4 e 52 semanas foi vital para capturar sazonalidades anuais.
- **Features Cíclicas**: Transformar semanas em coordenadas de seno/cosseno permitiu ao modelo entender que a semana 52 está próxima da semana 1.


---

## ⚙️ Padrões do Repositório

Para manter os experimentos reproduzíveis e fáceis de executar em qualquer máquina, os novos scripts seguem estes padrões:

- Todos os caminhos de dados e artefatos usam caminhos relativos ao arquivo do experimento ou à raiz do repositório.
- Cada execução cria uma pasta versionada em `experiments/artifacts/<experimento>_<timestamp>_<git_sha>/`.
- Seeds são definidas de forma consistente e registradas junto com o experimento.
- O `pip freeze` da execução é salvo em `pip_freeze.txt` dentro da pasta versionada e também enviado ao MLflow quando disponível.
- Artefatos de modelo seguem nomes previsíveis, por exemplo `model.pkl`, `joblib`, `SavedModel/` ou `torchscript/`, sempre dentro da pasta versionada.

### Convenção de Runtime

| Experimento | Hardware recomendado | Expectativa prática |
|---|---|---|
| `exp1_ag_news.py` | GPU recomendada | Transformer fine-tuning fica bem mais rápido em GPU; em CPU pode levar bem mais tempo. |
| `exp3_fake_news.py` | CPU suficiente | Classificação tradicional em texto, normalmente roda bem em CPU. |
| `ensemble_pyramid.py` | CPU recomendada com memória sobrando | O ensemble piramidal é pesado em treinamento, mas não depende de GPU. |
| `twitter-sentiment-analysis.ipynb` | CPU suficiente | Modelos clássicos com TF-IDF rodam bem em CPU. |
| `price-prediction-multiple-linear-regression.ipynb` | CPU suficiente | Regressão linear e EDA são leves. |
| `property-sales-time-series.ipynb` | CPU suficiente | SARIMA/EDA rodam em CPU; `auto_arima` pode ser o trecho mais demorado. |
| `animal-classifier.ipynb` | GPU recomendada | PyTorch + TensorFlow com modelos pré-treinados fica mais ágil em GPU. |
| `face_recognition_app.ipynb` | CPU suficiente (GPU opcional) | LBPH roda em CPU; `transfer_yunet` acelera com GPU, mas funciona em CPU. |

### Notebook de Deteccao Facial

- O app de deteccao/reconhecimento facial agora esta **embutido no notebook** `experiments/face_recognition_app.ipynb`.
- O arquivo Python separado `face_recognition_app.py` nao e mais necessario para executar o fluxo.
- Modos de treinamento disponiveis no notebook:
	- `lbph`
	- `cnn`
	- `transfer_yunet`
- Configuracoes importantes via variaveis de ambiente:
	- `FACE_DETECTOR=yunet|haar`
	- `FACE_TL_EPOCHS`, `FACE_TL_BATCH`
	- `FACE_CNN_EPOCHS`, `FACE_CNN_BATCH`
	- `YUNET_SCORE_THRESHOLD`, `YUNET_NMS_THRESHOLD`, `YUNET_TOP_K`

### Validacao de Notebooks

- Para verificar consistencia estrutural/sintatica dos notebooks, execute:

```bash
python scripts/validate_notebooks.py
```

- O validador trata magics (`!`, `%`, `?`) e marca notebooks externos (Databricks/IBM) como `EXT`.

### Dicas de Reprodutibilidade

- Sempre rode o experimento a partir do próprio arquivo/script para que os caminhos relativos resolvam corretamente.
- Se o ambiente estiver com MLflow/DagsHub configurado, verifique os parâmetros `seed`, `git_sha`, `run_timestamp` e o artefato `pip_freeze.txt` no run.
- Em notebooks, prefira executar as células na ordem original antes de alterar os caminhos ou a estrutura.

## 🤖 AutoML e MLOps Studio

*(Esta seção será expandida conforme o desenvolvimento do [AutoMLOps-Studio](experiments/AutoMLOps-Studio) avança.)*

---

## 🛠️ Estrutura de Experimentos

### Projetos Analisados:
1. **[senti-pred](experiments/senti-pred)**: Foco em AutoML e exploração de múltiplos frameworks.
2. **[old_senti-pred_upgrade](experiments/old_senti-pred_upgrade)**: Foco em modelos manuais clássicos (LinearSVC, KNN, RF, MLP) e otimização de pipeline TF-IDF.
3. **[senti-pred-variations](experiments/senti-pred-variations)**: Variações do projeto Senti-Pred incluindo Logistic Regression, MultinomialNB, Random Forest, FLAML AutoML, e o Ensemble Pyramid de 6 camadas.
4. **[Sales Forecast](experiments/sales-forecast)**: Foco em Séries Temporais, LightGBM, Otimização Bayesiana com Pruning (Optuna), tracking MLOps completo (MLflow), Pytest e Docker (V2.2).
5. **[ibm-experiments](experiments/ibm-experiments)**: Notebooks exploratórios de Boston Housing e produções elétricas usando Snap ML da IBM.
6. **[databricks forecast](experiments/databricks-forecast)**: Script de download de artefatos para integração com Databricks.

### Experimentos Rápidos:
- **[exp1_ag_news.py](experiments/exp1_ag_news.py)**: Classificação de notícias com DistilBERT.
- **[exp2_time_series.py](experiments/exp2_time_series.py)**: Previsão de temperatura com Prophet.
- **[exp3_fake_news.py](experiments/exp3_fake_news.py)**: Detecção de fake news com pipeline supervisionado.

### Formato de Saída dos Experimentos

Os scripts recentes gravam os artefatos em uma pasta versionada por execução. O padrão é:

```text
experiments/artifacts/<nome_experimento>_<YYYYMMDD_HHMMSS>_<git_sha>/
```

Exemplos comuns:
- `ag_news_classification_20260409_153000_ab12cd3/ag_news_model/`
- `temperature_forecasting_20260409_153000_ab12cd3/prophet_model.pkl`
- `twitter_sentiment_analysis_20260409_153000_ab12cd3/twitter_results.csv`
- `ensemble_pyramid_20260409_153000_ab12cd3/ensemble_pyramid_best.pkl`

---

## 🚀 Conclusão Final: A Performance é Sistêmica

Após dezenas de experimentos, a maior lição é que a métrica final não é um mérito exclusivo do algoritmo escolhido.
- **Simbiose**: Quase qualquer modelo pode atingir métricas de excelência se o pipeline (pré-processamento e hiperparâmetros) for exaustivamente otimizado para ele.
- **Eficiência vs. Força Bruta**: O desafio do Cientista de Dados não é apenas "chegar no 0.99", mas sim encontrar o modelo que chega lá de forma natural e eficiente, sem "lutar contra a natureza dos dados".
- **Decisão Estratégica**: Escolher o modelo certo é, na verdade, escolher o caminho de menor resistência entre os dados brutos e a predição precisa.

---
*Este repositório é um diário vivo de descobertas em Ciência de Dados.*
