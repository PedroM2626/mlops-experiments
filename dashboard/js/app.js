/* ============================================================
   MLOps Experiments Dashboard - Application Logic
   All experiment data + rendering + filters + search + navigation
   ============================================================ */

const App = (() => {

  /* -------------------------------------------------------
     EXPERIMENT DATA - All 38 experiments
     ------------------------------------------------------- */
  const experiments = [
    // --- NLP - Analise de Sentimento (Senti-Pred) ---
    {
      id: 1,
      title: 'Senti-Pred Baseline (RoBERTa)',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Primeiro experimento usando Transformer pre-treinado (RoBERTa) para classificacao de sentimentos em tweets. Amostragem de apenas 1k linhas demonstrou as limitacoes de modelos grandes com poucos dados.',
      techniques: ['RoBERTa', 'Transformer', 'Fine-tuning', 'Transfer Learning'],
      metric: { label: 'F1-Macro', value: '~60%', percent: 60 },
      script: 'experiments/senti-pred-variations/senti-pred-exp1/',
      models: ['RoBERTa (Pre-trained)'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'Lento e com pouco dado para o modelo. Demonstrou que Transformers exigem hardware e dados massivos.'
    },
    {
      id: 2,
      title: 'Senti-Pred Classic (Logistic Regression)',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Salto enorme usando o dataset completo com TF-IDF e Logistic Regression. Primeiro modelo robusto com o dataset total.',
      techniques: ['TF-IDF', 'Logistic Regression', 'N-grams'],
      metric: { label: 'Accuracy', value: '87.2%', percent: 87.2 },
      script: 'experiments/senti-pred-variations/Senti-Pred-Remake/tran_classic.py',
      models: ['Logistic Regression'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'TF-IDF com 10k features. Primeiro modelo robusto treinado no dataset completo.'
    },
    {
      id: 3,
      title: 'Senti-Pred Optimized (LR + Regex)',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'O poder da limpeza de texto (noise removal). Salto de 87% para 95% apenas com melhor pre-processamento usando Regex.',
      techniques: ['TF-IDF', 'Logistic Regression', 'Regex Cleaning', 'Noise Removal'],
      metric: { label: 'Accuracy', value: '95.3%', percent: 95.3 },
      script: 'experiments/senti-pred-variations/Senti-Pred-Remake/train_optimized.py',
      models: ['Logistic Regression'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'TF-IDF com 20k features + limpeza de texto com Regex (URLs, mencoes, caracteres repetidos).'
    },
    {
      id: 4,
      title: 'Senti-Pred Ultimate (Passive Aggressive)',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Algoritmo mais agressivo para correcao de erros. Foco em correcao agressiva de erros de classificacao com Passive Aggressive Classifier.',
      techniques: ['TF-IDF', 'Passive Aggressive', 'Char Repetition Fix'],
      metric: { label: 'Accuracy', value: '97.0%', percent: 97.0 },
      script: 'experiments/senti-pred-variations/Senti-Pred-Remake/train_ultimate.py',
      models: ['Passive Aggressive Classifier'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'TF-IDF com 40k features + correcao de caracteres repetidos ("loooove" -> "love").'
    },
    {
      id: 5,
      title: 'Senti-Pred God Mode (Voting Ensemble)',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Recorde do Remake 1. Voting Classifier combinando Passive Aggressive e Logistic Regression eliminou erros individuais com "democracia" entre modelos.',
      techniques: ['TF-IDF', 'Voting Ensemble', 'Passive Aggressive', 'LogReg'],
      metric: { label: 'Accuracy', value: '97.5%', percent: 97.5 },
      script: 'experiments/senti-pred-variations/Senti-Pred-Remake/train_god_mode.py',
      models: ['Voting Classifier (PA + LR)'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'TF-IDF com 50k features + preservacao de pontuacao emocional (!, ?). Recorde consolidado do Remake 1.'
    },
    {
      id: 6,
      title: 'Senti-Pred Insane (Stacking)',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Queda por overfitting. Stacking Classifier com Chi2 Feature Selection mostrou que excesso de complexidade reduz performance.',
      techniques: ['TF-IDF', 'Stacking Classifier', 'Chi2', 'Feature Selection'],
      metric: { label: 'Accuracy', value: '96.2%', percent: 96.2 },
      script: 'experiments/senti-pred-variations/Senti-Pred-Remake/train_insane_mode.py',
      models: ['Stacking Classifier'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'Overfitting leve. Demonstrou que a complexidade nem sempre significa melhora.'
    },
    {
      id: 7,
      title: 'Senti-Pred Remake2 (Recorde Absoluto)',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Recorde absoluto do projeto. Voting Ensemble (LinearSVC + LR) com TF-IDF de 100k features e 4-grams capturando nuances contextuais.',
      techniques: ['TF-IDF 100k', '4-grams', 'LinearSVC', 'Voting Ensemble'],
      metric: { label: 'Accuracy', value: '97.80%', percent: 97.8 },
      script: 'experiments/senti-pred-variations/Senti-Pred-remake2/',
      models: ['Voting (LinearSVC + LogReg)'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'Recorde absoluto do projeto inteiro. 100k features com 4-grams para capturar nuances.'
    },
    {
      id: 8,
      title: 'MultinomialNB Senti-Pred',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Naive Bayes otimizado via busca logaritmica de Alpha. Trigramas com sublinear TF para classificacao de sentimentos.',
      techniques: ['TF-IDF', 'MultinomialNB', 'Trigrams', 'Sublinear TF'],
      metric: { label: 'Accuracy', value: '92.06%', percent: 92.06 },
      script: 'experiments/senti-pred-variations/multinomialnb-Senti-Pred/',
      models: ['Multinomial Naive Bayes'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'Otimizado via busca logaritmica de Alpha para encontrar o melhor suavizamento.'
    },
    {
      id: 9,
      title: 'Random Forest Senti-Pred',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Salto dramatico de 71% para 91% apos otimizacao de hiperparametros com Optuna (busca profunda).',
      techniques: ['TF-IDF', 'Random Forest', 'Optuna', 'HPO'],
      metric: { label: 'Accuracy', value: '91.0%', percent: 91.0 },
      script: 'experiments/senti-pred-variations/random_forest-Senti-Pred/',
      models: ['Random Forest Classifier'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'Salto de 71% -> 91% apos HPO com Optuna. Demonstrou o impacto da otimizacao.'
    },
    {
      id: 10,
      title: 'Logistic Regression Senti-Pred',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Baseline linear extremamente estavel. TF-IDF com 20k features e limpeza Regex para classificacao robusta.',
      techniques: ['TF-IDF', 'Logistic Regression', 'Regex'],
      metric: { label: 'Accuracy', value: '96.0%', percent: 96.0 },
      script: 'experiments/senti-pred-variations/logistic-senti-pred/',
      models: ['Logistic Regression'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'Baseline linear extremamente estavel com TF-IDF 20k features.'
    },
    {
      id: 11,
      title: 'FLAML AutoML Senti-Pred',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Melhor resultado AutoML. FLAML selecionou automaticamente RandomForest como vencedor em apenas 5 minutos de busca.',
      techniques: ['FLAML', 'AutoML', 'TF-IDF', 'Auto Feature Selection'],
      metric: { label: 'Accuracy', value: '96.73%', percent: 96.73 },
      script: 'experiments/senti-pred-variations/flaml-Senti-Pred/',
      models: ['FLAML -> RandomForest (auto-selected)'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'TF-IDF 30k + 1-2 n-grams. RandomForest selecionado como vencedor em 300s.'
    },
    {
      id: 12,
      title: 'AutoGluon Senti-Pred',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Experimentacao com framework AutoGluon da Amazon para AutoML. Requer Python 3.11 dedicado.',
      techniques: ['AutoGluon', 'AutoML', 'Ensemble'],
      metric: null,
      script: 'experiments/senti-pred-variations/autogluon_senti_pred.py',
      models: ['AutoGluon Auto-selected Models'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'Framework AutoGluon nao possui wheel para Python 3.14. Usa .venv311.'
    },
    {
      id: 13,
      title: 'Ensemble Pyramid (6 Camadas)',
      category: 'nlp-sentiment',
      categoryLabel: 'NLP - Analise de Sentimento',
      status: 'completed',
      description: 'Arquitetura piramidal com 6 camadas de ensembles hierarquicos. Combina Bagging, Voting e Stacking com RL Meta-Learner para otimizacao automatica.',
      techniques: ['Bagging', 'Voting', 'Stacking', 'RL Meta-Learner', 'Thompson Sampling', 'TF-IDF 70k'],
      metric: { label: 'F1-Score', value: '~98%+', percent: 98 },
      script: 'experiments/ensemble_pyramid.py',
      models: ['LR', 'LinearSVC', 'NB', 'CNB', 'Ridge', 'RF', 'ExtraTrees', 'Meta-Stacking', 'Meta-Voting'],
      dataset: 'Twitter Sentiment (4 classes)',
      details: 'Motor AutoML com RL que decide quantos e quais modelos usar por camada. Skip connections, jitter de hiperparametros, early stopping. CLI customizavel: --layers, --epsilon, --strategy (dense/residual/simple).'
    },
    // --- NLP - Classificacao e Extracao ---
    {
      id: 14,
      title: 'AG News Classification (DistilBERT)',
      category: 'nlp-classification',
      categoryLabel: 'NLP - Classificacao e Extracao',
      status: 'completed',
      description: 'Classificacao de noticias em 4 categorias usando Transfer Learning com DistilBERT. Demonstrou que Transformers pre-treinados superam TF-IDF para tarefas semanticas complexas.',
      techniques: ['DistilBERT', 'Fine-tuning', 'Transfer Learning', 'Transformers'],
      metric: { label: 'Accuracy', value: 'Alta', percent: 92 },
      script: 'experiments/exp1_ag_news.py',
      models: ['DistilBERT (Fine-tuned)'],
      dataset: 'AG News (4 categorias)',
      details: 'GPU recomendada. Transformer fine-tuning com classificacao de World, Sports, Business, Sci/Tech.'
    },
    {
      id: 15,
      title: 'Fake News Detection',
      category: 'nlp-classification',
      categoryLabel: 'NLP - Classificacao e Extracao',
      status: 'completed',
      description: 'Pipeline completo de deteccao de fake news com TF-IDF, features linguisticas e Ensemble Voting. F1 perfeito de 1.000.',
      techniques: ['TF-IDF', 'Logistic Regression', 'Random Forest', 'Ensemble Voting'],
      metric: { label: 'F1-Score', value: '1.000', percent: 100 },
      script: 'experiments/exp3_fake_news.py',
      models: ['TF-IDF + LR', 'Linguistic Features + RF', 'Ensemble Voting'],
      dataset: 'Fake News Dataset',
      details: 'Execucao em ~5s. MLflow tracked. Artefatos em experiments/artifacts/fake_news_detection/.'
    },
    {
      id: 16,
      title: 'Clustering + Topic Modeling',
      category: 'nlp-classification',
      categoryLabel: 'NLP - Classificacao e Extracao',
      status: 'partial',
      description: 'Agrupamento de textos e modelagem de topicos usando K-Means e LDA. Bloqueado por erro de importacao do gensim.',
      techniques: ['K-Means', 'LDA', 'Gensim', 'Topic Modeling'],
      metric: null,
      script: 'experiments/exp5_clustering_topics.py',
      models: ['K-Means', 'LDA (Gensim)'],
      dataset: 'Text Corpus',
      details: 'Bloqueador: gensim import error. Solucao: pip install gensim.'
    },
    {
      id: 17,
      title: 'NER + Information Extraction',
      category: 'nlp-classification',
      categoryLabel: 'NLP - Classificacao e Extracao',
      status: 'partial',
      description: 'Extracao de entidades nomeadas (PERSON, ORG, LOCATION, DATE, MONEY, PERCENT) usando spaCy. Bloqueado por erros de DataFrame.',
      techniques: ['spaCy', 'NER', 'Information Extraction'],
      metric: null,
      script: 'experiments/exp6_ner_extraction.py',
      models: ['spaCy NER Pipeline'],
      dataset: 'Text Documents',
      details: 'Entidades suportadas: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PERCENT. Bloqueador: DataFrame access errors.'
    },
    {
      id: 18,
      title: 'Multi-Task Learning',
      category: 'nlp-classification',
      categoryLabel: 'NLP - Classificacao e Extracao',
      status: 'blocked',
      description: 'Aprendizado multi-tarefa combinando classificacao de sentimento, predicao de intensidade e classificacao de topicos em uma unica rede.',
      techniques: ['Multi-Task', 'Sentiment', 'Intensity Prediction', 'Topic Classification'],
      metric: null,
      script: 'experiments/exp7_multitask_learning.py',
      models: ['Multi-Task Neural Network'],
      dataset: 'Multi-label Dataset',
      details: 'Bloqueado por erros de sintaxe (escape sequences). 3 tarefas simultaneas.'
    },
    {
      id: 19,
      title: 'Twitter Sentiment Analysis (Notebook)',
      category: 'nlp-classification',
      categoryLabel: 'NLP - Classificacao e Extracao',
      status: 'completed',
      description: 'Analise de sentimento em tweets usando modelos classicos com TF-IDF. Notebook exploratoria com EDA e modelagem.',
      techniques: ['TF-IDF', 'Classical ML', 'EDA'],
      metric: null,
      script: 'experiments/twitter-sentiment-analysis.ipynb',
      models: ['Classical ML Models'],
      dataset: 'Twitter Dataset',
      details: 'Notebook com analise exploratoria e modelos classicos. CPU suficiente.'
    },
    // --- Series Temporais e Forecast ---
    {
      id: 20,
      title: 'Time Series - Prophet (Temperatura)',
      category: 'timeseries',
      categoryLabel: 'Series Temporais e Forecast',
      status: 'completed',
      description: 'Previsao de temperaturas diarias usando Facebook Prophet. Captura automatica de sazonalidades diaria, semanal e anual.',
      techniques: ['Prophet', 'Seasonality', 'Time Series'],
      metric: null,
      script: 'experiments/exp2_time_series.py',
      models: ['Facebook Prophet'],
      dataset: 'Daily Minimum Temperatures',
      details: 'Prophet excelente para capturar sazonalidades de forma automatica e robusta a feriados.'
    },
    {
      id: 21,
      title: 'Sales Forecast (Hackathon LightGBM)',
      category: 'timeseries',
      categoryLabel: 'Series Temporais e Forecast',
      status: 'completed',
      description: 'Previsao de vendas com LightGBM otimizado via Optuna (100 trials de busca Bayesiana). Engenharia de features com lags, rolling windows e features ciclicas.',
      techniques: ['LightGBM', 'Optuna', 'Feature Engineering', 'Lags', 'Rolling Windows'],
      metric: { label: 'MAE', value: '2.576', percent: 85 },
      script: 'experiments/sales forecast/scripts/',
      models: ['LightGBM (Optuna-optimized)'],
      dataset: 'Sales Transactions 2022',
      details: '20+ features criadas. Lags sazonais, janelas moveis, features ciclicas (seno/cosseno). Validacao hold-out temporal.'
    },
    {
      id: 22,
      title: 'Property Sales Time Series (SARIMA)',
      category: 'timeseries',
      categoryLabel: 'Series Temporais e Forecast',
      status: 'completed',
      description: 'Analise de series temporais de vendas imobiliarias usando SARIMA e auto_arima. Notebook exploratoria com EDA e previsoes.',
      techniques: ['SARIMA', 'auto_arima', 'EDA', 'Decomposition'],
      metric: null,
      script: 'experiments/property-sales-time-series.ipynb',
      models: ['SARIMA', 'auto_arima'],
      dataset: 'Property Sales',
      details: 'CPU suficiente. auto_arima pode ser o trecho mais demorado.'
    },
    {
      id: 23,
      title: 'Knowledge Distillation - Time Series',
      category: 'timeseries',
      categoryLabel: 'Series Temporais e Forecast',
      status: 'completed',
      description: 'Destilacao de conhecimento aplicada a series temporais. Transferencia de conhecimento de modelo grande (teacher) para modelo menor (student).',
      techniques: ['Knowledge Distillation', 'Teacher-Student', 'Time Series'],
      metric: null,
      script: 'experiments/knowledge_distillation-time_series.ipynb',
      models: ['Teacher Model', 'Student Model (Distilled)'],
      dataset: 'Time Series Dataset',
      details: 'Tecnica de compressao de modelos para deploy eficiente.'
    },
    {
      id: 24,
      title: 'Databricks Forecast (DeepAR / Prophet)',
      category: 'timeseries',
      categoryLabel: 'Series Temporais e Forecast',
      status: 'external',
      description: 'Forecasting gerenciado pela plataforma Databricks usando DeepAR e Prophet. Inclui scripts de download de artefatos.',
      techniques: ['DeepAR', 'Prophet', 'Databricks', 'Cloud ML'],
      metric: null,
      script: 'experiments/databricks forecast/',
      models: ['DeepAR', 'Prophet (Databricks)'],
      dataset: 'Sales Quantity Transactions',
      details: 'Notebooks Databricks com preprocessing, training e tuning. Inclui download_artifacts.py.'
    },
    {
      id: 25,
      title: 'Electric Production Forecast (IBM)',
      category: 'timeseries',
      categoryLabel: 'Series Temporais e Forecast',
      status: 'completed',
      description: 'Previsao de producao eletrica usando Snap ML da IBM Watson Studio. Inclui variacao com montador.',
      techniques: ['Snap ML', 'IBM Watson', 'Time Series'],
      metric: null,
      script: 'experiments/ibm-experiments/Electric_Production.ipynb',
      models: ['Snap ML Forecaster'],
      dataset: 'Electric Production',
      details: 'Notebook IBM Watson Studio. Inclui variacao P4 - Montador.'
    },
    // --- Computer Vision ---
    {
      id: 26,
      title: 'Animal Classifier (PyTorch + TF)',
      category: 'cv',
      categoryLabel: 'Computer Vision',
      status: 'completed',
      description: 'Classificador de animais usando Transfer Learning com modelos pre-treinados em PyTorch e TensorFlow.',
      techniques: ['PyTorch', 'TensorFlow', 'Transfer Learning', 'CNN', 'Image Classification'],
      metric: null,
      script: 'experiments/animal-classifier.ipynb',
      models: ['Pre-trained CNNs (PyTorch)', 'Pre-trained CNNs (TensorFlow)'],
      dataset: 'Animal Images',
      details: 'GPU recomendada. Usa modelos pre-treinados de ambos frameworks.'
    },
    {
      id: 27,
      title: 'Face Recognition App',
      category: 'cv',
      categoryLabel: 'Computer Vision',
      status: 'completed',
      description: 'Aplicacao de deteccao e reconhecimento facial com 3 modos de treinamento: LBPH, CNN e Transfer Learning com YuNet.',
      techniques: ['LBPH', 'CNN', 'YuNet', 'Face Detection', 'Face Recognition'],
      metric: null,
      script: 'experiments/face_recognition_app.ipynb',
      models: ['LBPH Recognizer', 'CNN', 'YuNet Transfer Learning'],
      dataset: 'Face Images',
      details: 'Detectores: yunet, haar. Configs via env: FACE_DETECTOR, FACE_TL_EPOCHS, YUNET_SCORE_THRESHOLD.'
    },
    {
      id: 28,
      title: 'Image Recommender',
      category: 'cv',
      categoryLabel: 'Computer Vision',
      status: 'completed',
      description: 'Sistema de recomendacao de imagens baseado em similaridade visual. Usa embeddings de CNNs pre-treinadas.',
      techniques: ['Image Similarity', 'Embeddings', 'CNN Features', 'Recommendation'],
      metric: null,
      script: 'experiments/image_recommender.ipynb',
      models: ['CNN Feature Extractor'],
      dataset: 'Image Dataset',
      details: 'Recomendacao baseada em distancia entre feature vectors de CNNs pre-treinadas.'
    },
    {
      id: 29,
      title: 'YOLO Object Detection',
      category: 'cv',
      categoryLabel: 'Computer Vision',
      status: 'completed',
      description: 'Deteccao de objetos em tempo real usando YOLO (You Only Look Once). Notebook exploratoria.',
      techniques: ['YOLO', 'Object Detection', 'Real-time'],
      metric: null,
      script: 'experiments/yolo_notebook.ipynb',
      models: ['YOLO'],
      dataset: 'Object Detection Dataset',
      details: 'Deteccao de objetos em tempo real com bounding boxes.'
    },
    // --- Anomalias, Trading e Monitoramento ---
    {
      id: 30,
      title: 'Anomaly Detection (Multi-Method)',
      category: 'anomaly',
      categoryLabel: 'Anomalias, Trading e Monitoramento',
      status: 'completed',
      description: 'Deteccao de anomalias com 5 metodos diferentes: Z-Score, Isolation Forest, LOF, Elliptic Envelope e Prophet. Testado em 3 datasets.',
      techniques: ['Z-Score', 'Isolation Forest', 'LOF', 'Elliptic Envelope', 'Prophet'],
      metric: null,
      script: 'experiments/exp4_anomaly_detection.py',
      models: ['Z-Score', 'Isolation Forest', 'Local Outlier Factor', 'Elliptic Envelope', 'Prophet'],
      dataset: '3 Datasets de Anomalia',
      details: 'Execucao em ~8s. MLflow tracked. 5 metodos comparados em 3 datasets.'
    },
    {
      id: 31,
      title: 'RL Trading (Q-Learning)',
      category: 'anomaly',
      categoryLabel: 'Anomalias, Trading e Monitoramento',
      status: 'completed',
      description: 'Agente de Reinforcement Learning para trading usando Q-Learning. Compara retorno do RL Agent vs Buy & Hold.',
      techniques: ['Q-Learning', 'Reinforcement Learning', 'Trading'],
      metric: { label: 'RL vs B&H', value: '0% vs 38.39%', percent: 30 },
      script: 'experiments/exp9_rl_trading.py',
      models: ['Q-Learning Agent'],
      dataset: 'Market Data',
      details: '20 episodios de treinamento. RL Agent: +0.00% vs Buy & Hold: +38.39%. Execucao em ~3s.'
    },
    {
      id: 32,
      title: 'Data Drift Monitoring (KS Test)',
      category: 'anomaly',
      categoryLabel: 'Anomalias, Trading e Monitoramento',
      status: 'completed',
      description: 'Monitoramento de data drift usando o teste Kolmogorov-Smirnov. 100% de acuracia de deteccao em 2 cenarios.',
      techniques: ['Kolmogorov-Smirnov', 'Data Drift', 'Statistical Testing'],
      metric: { label: 'Detection', value: '100%', percent: 100 },
      script: 'experiments/exp10_drift_monitoring.py',
      models: ['KS Test Statistical'],
      dataset: '2 Cenarios de Drift',
      details: 'Execucao em ~2s. MLflow tracked. 100% acuracia de deteccao.'
    },
    {
      id: 33,
      title: 'Model Explainability & Feature Importance',
      category: 'anomaly',
      categoryLabel: 'Anomalias, Trading e Monitoramento',
      status: 'completed',
      description: 'Explicabilidade de modelos e importancia de features usando SHAP e tecnicas de feature importance. 15 top features identificadas.',
      techniques: ['SHAP', 'Feature Importance', 'Explainability', 'XAI'],
      metric: { label: 'Model Acc', value: '100%', percent: 100 },
      script: 'experiments/exp11_explainability_final.py',
      models: ['Logistic Regression', 'Random Forest'],
      dataset: 'Classification Dataset',
      details: '15 top features identificadas. 2 modelos analisados. Execucao em ~4s.'
    },
    // --- IBM Watson Studio ---
    {
      id: 34,
      title: 'Boston Housing Price Prediction',
      category: 'ibm',
      categoryLabel: 'IBM Watson Studio',
      status: 'completed',
      description: 'Predicao de precos imobiliarios em Boston usando Snap ML Regression da IBM Watson Studio.',
      techniques: ['Snap ML', 'Regression', 'IBM Watson'],
      metric: null,
      script: 'experiments/ibm-experiments/Boston Housing Price Prediction.ipynb',
      models: ['Snap ML Regressor'],
      dataset: 'Boston Housing',
      details: 'Notebook IBM Watson Studio com Snap ML para regressao rapida.'
    },
    {
      id: 35,
      title: 'Classificador RF Snap ML',
      category: 'ibm',
      categoryLabel: 'IBM Watson Studio',
      status: 'completed',
      description: 'Classificador de floresta aleatoria usando Snap ML da IBM. P5 com Random Forest para classificacao.',
      techniques: ['Snap ML', 'Random Forest', 'Classification'],
      metric: null,
      script: 'experiments/ibm-experiments/_P5 - Classificador de floresta aleatória de Sn....ipynb',
      models: ['Snap ML Random Forest Classifier'],
      dataset: 'Classification Dataset (IBM)',
      details: 'Notebook P5 IBM Watson Studio.'
    },
    {
      id: 36,
      title: 'Regressor RF Snap ML',
      category: 'ibm',
      categoryLabel: 'IBM Watson Studio',
      status: 'completed',
      description: 'Regressor de floresta aleatoria usando Snap ML da IBM. P5 com Random Forest para regressao.',
      techniques: ['Snap ML', 'Random Forest', 'Regression'],
      metric: null,
      script: 'experiments/ibm-experiments/_P5 - Regressor de floresta aleatória de Snap_ ....ipynb',
      models: ['Snap ML Random Forest Regressor'],
      dataset: 'Regression Dataset (IBM)',
      details: 'Notebook P5 IBM Watson Studio.'
    },
    {
      id: 37,
      title: 'Analise de Sentimentos (IBM)',
      category: 'ibm',
      categoryLabel: 'IBM Watson Studio',
      status: 'completed',
      description: 'Analise de sentimentos usando Snap ML da IBM Watson Studio. Classificacao de textos.',
      techniques: ['Snap ML', 'Sentiment Analysis', 'NLP'],
      metric: null,
      script: 'experiments/ibm-experiments/analise_de_sentimentos.ipynb',
      models: ['Snap ML Classifier'],
      dataset: 'Sentiment Dataset (IBM)',
      details: 'Notebook IBM Watson Studio para analise de sentimentos.'
    },
    // --- Regressao ---
    {
      id: 38,
      title: 'Price Prediction (Multiple Linear Regression)',
      category: 'regression',
      categoryLabel: 'Regressao',
      status: 'completed',
      description: 'Predicao de precos usando Regressao Linear Multipla. Notebook com EDA completa e modelagem.',
      techniques: ['Multiple Linear Regression', 'EDA', 'Feature Analysis'],
      metric: null,
      script: 'experiments/price-prediction-multiple-linear-regression.ipynb',
      models: ['Multiple Linear Regression'],
      dataset: 'Price Dataset',
      details: 'CPU suficiente. Regressao linear e EDA sao leves.'
    },
  ];

  /* -------------------------------------------------------
     Category metadata
     ------------------------------------------------------- */
  const categories = {
    'all': { label: 'Todos', icon: '\u{1F3AF}', color: '#818cf8' },
    'nlp-sentiment': { label: 'NLP - Sentimento', icon: '\u{1F4AC}', color: '#f472b6' },
    'nlp-classification': { label: 'NLP - Classificacao', icon: '\u{1F4F0}', color: '#fb923c' },
    'timeseries': { label: 'Series Temporais', icon: '\u{1F4C8}', color: '#34d399' },
    'cv': { label: 'Computer Vision', icon: '\u{1F441}', color: '#38bdf8' },
    'anomaly': { label: 'Anomalias & Monitoring', icon: '\u{1F50D}', color: '#a78bfa' },
    'ibm': { label: 'IBM Watson Studio', icon: '\u{2601}', color: '#fbbf24' },
    'regression': { label: 'Regressao', icon: '\u{1F4CA}', color: '#f87171' },
  };

  /* -------------------------------------------------------
     Senti-Pred Evolution data (for timeline chart)
     ------------------------------------------------------- */
  const sentipredEvolution = [
    { label: 'RoBERTa', value: 60 },
    { label: 'Classic', value: 87.2 },
    { label: 'RF', value: 91 },
    { label: 'MNB', value: 92.06 },
    { label: 'Optimized', value: 95.3 },
    { label: 'Logistic', value: 96 },
    { label: 'Insane', value: 96.2 },
    { label: 'FLAML', value: 96.73 },
    { label: 'Ultimate', value: 97 },
    { label: 'God Mode', value: 97.5 },
    { label: 'Remake2', value: 97.8, best: true },
    { label: 'Pyramid', value: 98, best: true },
  ];

  /* -------------------------------------------------------
     State
     ------------------------------------------------------- */
  let state = {
    activeCategory: 'all',
    searchQuery: '',
    statusFilter: 'all',
    sidebarOpen: false,
  };

  /* -------------------------------------------------------
     Computed
     ------------------------------------------------------- */
  function getFilteredExperiments() {
    return experiments.filter(exp => {
      const matchesCategory = state.activeCategory === 'all' || exp.category === state.activeCategory;
      const matchesStatus = state.statusFilter === 'all' || exp.status === state.statusFilter;
      const matchesSearch = state.searchQuery === '' ||
        exp.title.toLowerCase().includes(state.searchQuery.toLowerCase()) ||
        exp.description.toLowerCase().includes(state.searchQuery.toLowerCase()) ||
        exp.techniques.some(t => t.toLowerCase().includes(state.searchQuery.toLowerCase())) ||
        exp.categoryLabel.toLowerCase().includes(state.searchQuery.toLowerCase());
      return matchesCategory && matchesStatus && matchesSearch;
    });
  }

  function getStats() {
    const total = experiments.length;
    const completed = experiments.filter(e => e.status === 'completed').length;
    const categoriesCount = new Set(experiments.map(e => e.category)).size;
    const techniques = new Set(experiments.flatMap(e => e.techniques)).size;
    return { total, completed, categoriesCount, techniques };
  }

  function getCategoryStats() {
    const counts = {};
    experiments.forEach(exp => {
      counts[exp.category] = (counts[exp.category] || 0) + 1;
    });
    return counts;
  }

  /* -------------------------------------------------------
     RENDER FUNCTIONS
     ------------------------------------------------------- */

  function renderSidebar() {
    const catStats = getCategoryStats();
    const stats = getStats();

    const navItems = Object.entries(categories).map(([key, cat]) => {
      const count = key === 'all' ? experiments.length : (catStats[key] || 0);
      const activeClass = state.activeCategory === key ? 'active' : '';
      return `<div class="sidebar-nav-item ${activeClass}" data-category="${key}" onclick="App.setCategory('${key}')">
        <span class="icon">${cat.icon}</span>
        <span>${cat.label}</span>
        <span class="count">${count}</span>
      </div>`;
    }).join('');

    return `
      <div class="sidebar-header">
        <div class="sidebar-logo">
          <div class="sidebar-logo-icon">\u{1F9EA}</div>
          <div class="sidebar-logo-text">
            <h2>MLOps Lab</h2>
            <span>Experiments Hub</span>
          </div>
        </div>
      </div>
      <nav class="sidebar-nav">
        <div class="sidebar-section-title">Categorias</div>
        ${navItems}
      </nav>
      <div class="sidebar-stats">
        <div class="sidebar-stats-grid">
          <div class="sidebar-stat">
            <div class="value">${stats.total}</div>
            <div class="label">Experimentos</div>
          </div>
          <div class="sidebar-stat">
            <div class="value">${stats.completed}</div>
            <div class="label">Completos</div>
          </div>
          <div class="sidebar-stat">
            <div class="value">${stats.categoriesCount}</div>
            <div class="label">Categorias</div>
          </div>
          <div class="sidebar-stat">
            <div class="value">${stats.techniques}</div>
            <div class="label">Tecnicas</div>
          </div>
        </div>
      </div>
    `;
  }

  function renderTopbar() {
    const catLabel = categories[state.activeCategory]?.label || 'Todos';
    const filtered = getFilteredExperiments();

    return `
      <button class="mobile-menu-btn" onclick="App.toggleSidebar()">\u2630</button>
      <div class="topbar-title">${catLabel} <span>${filtered.length} experimentos</span></div>
      <div class="search-container">
        <span class="search-icon">\u{1F50E}</span>
        <input type="text" class="search-input" placeholder="Buscar experimentos, tecnicas, modelos..."
          value="${state.searchQuery}" oninput="App.setSearch(this.value)" />
      </div>
      <div class="topbar-filters">
        <button class="filter-btn ${state.statusFilter === 'all' ? 'active' : ''}" onclick="App.setStatus('all')">Todos</button>
        <button class="filter-btn ${state.statusFilter === 'completed' ? 'active' : ''}" onclick="App.setStatus('completed')">Completos</button>
        <button class="filter-btn ${state.statusFilter === 'partial' ? 'active' : ''}" onclick="App.setStatus('partial')">Parciais</button>
        <button class="filter-btn ${state.statusFilter === 'blocked' ? 'active' : ''}" onclick="App.setStatus('blocked')">Bloqueados</button>
      </div>
    `;
  }

  function renderOverview() {
    const stats = getStats();
    const completionPct = Math.round((stats.completed / stats.total) * 100);

    return `
      <div class="overview-section">
        <div class="overview-header">
          <div>
            <h1>Repositorio de <span class="gradient-text">Experimentos MLOps</span></h1>
            <p>Jornada de aprendizado em Machine Learning, NLP, Computer Vision, Series Temporais e mais</p>
          </div>
        </div>
        <div class="stats-row">
          <div class="stat-card animate-in animate-in-1">
            <div class="stat-icon" style="background: rgba(129,140,248,0.15); color: #818cf8;">\u{1F9EA}</div>
            <div class="stat-value">${stats.total}</div>
            <div class="stat-label">Experimentos</div>
            <div class="stat-change positive">${completionPct}% completos</div>
          </div>
          <div class="stat-card animate-in animate-in-2">
            <div class="stat-icon" style="background: rgba(52,211,153,0.15); color: #34d399;">\u2714</div>
            <div class="stat-value">${stats.completed}</div>
            <div class="stat-label">Completos</div>
          </div>
          <div class="stat-card animate-in animate-in-3">
            <div class="stat-icon" style="background: rgba(244,114,182,0.15); color: #f472b6;">\u{1F4E6}</div>
            <div class="stat-value">${stats.categoriesCount}</div>
            <div class="stat-label">Categorias</div>
          </div>
          <div class="stat-card animate-in animate-in-4">
            <div class="stat-icon" style="background: rgba(251,191,36,0.15); color: #fbbf24;">\u{1F527}</div>
            <div class="stat-value">${stats.techniques}</div>
            <div class="stat-label">Tecnicas Unicas</div>
          </div>
          <div class="stat-card animate-in animate-in-5">
            <div class="stat-icon" style="background: rgba(167,139,250,0.15); color: #a78bfa;">\u{1F3C6}</div>
            <div class="stat-value">97.8%</div>
            <div class="stat-label">Melhor Accuracy (Senti-Pred)</div>
          </div>
        </div>
      </div>
    `;
  }

  function renderCharts() {
    return `
      <div class="charts-section">
        <div class="chart-card animate-in animate-in-1">
          <h3>Distribuicao por Categoria</h3>
          <p class="chart-subtitle">Numero de experimentos por area de ML</p>
          <div class="chart-canvas-wrap">
            <canvas id="chart-categories"></canvas>
          </div>
        </div>
        <div class="chart-card animate-in animate-in-2">
          <h3>Status dos Experimentos</h3>
          <p class="chart-subtitle">Progresso geral do repositorio</p>
          <div class="chart-canvas-wrap">
            <canvas id="chart-status"></canvas>
          </div>
        </div>
        <div class="chart-card animate-in animate-in-3">
          <h3>Evolucao Senti-Pred</h3>
          <p class="chart-subtitle">Accuracy ao longo dos experimentos de sentimento</p>
          <div class="chart-canvas-wrap">
            <canvas id="chart-evolution"></canvas>
          </div>
        </div>
        <div class="chart-card animate-in animate-in-4">
          <h3>Radar de Tecnicas</h3>
          <p class="chart-subtitle">Distribuicao por tipo de abordagem</p>
          <div class="chart-canvas-wrap">
            <canvas id="chart-radar"></canvas>
          </div>
        </div>
      </div>
    `;
  }

  function renderExperimentCard(exp, index) {
    const statusLabels = {
      completed: 'Completo',
      partial: 'Parcial',
      blocked: 'Bloqueado',
      external: 'Externo',
    };

    const metricHtml = exp.metric ? `
      <div class="card-metric">
        <span class="metric-label">${exp.metric.label}</span>
        <div class="metric-bar"><div class="metric-bar-fill" style="width: ${exp.metric.percent}%"></div></div>
        <span class="metric-value">${exp.metric.value}</span>
      </div>
    ` : '';

    const tags = exp.techniques.slice(0, 4).map((t, i) =>
      `<span class="tag ${i === 0 ? 'highlight' : ''}">${t}</span>`
    ).join('');

    const delay = Math.min(index, 8);

    return `
      <div class="experiment-card animate-in animate-in-${delay % 6 + 1}" data-category="${exp.category}" onclick="App.showDetail(${exp.id})">
        <div class="card-top">
          <span class="card-number">#${String(exp.id).padStart(2, '0')}</span>
          <span class="status-badge ${exp.status}">${statusLabels[exp.status]}</span>
        </div>
        <h3 class="card-title">${exp.title}</h3>
        <p class="card-description">${exp.description}</p>
        <div class="card-tags">${tags}</div>
        ${metricHtml}
      </div>
    `;
  }

  function renderExperiments() {
    const filtered = getFilteredExperiments();

    if (filtered.length === 0) {
      return `
        <div class="no-results">
          <div class="no-results-icon">\u{1F50D}</div>
          <h3>Nenhum experimento encontrado</h3>
          <p>Tente ajustar os filtros ou o termo de busca</p>
        </div>
      `;
    }

    // Group by category
    if (state.activeCategory === 'all' && state.searchQuery === '' && state.statusFilter === 'all') {
      return renderGroupedByCategory(filtered);
    }

    return `<div class="experiments-grid">${filtered.map((exp, i) => renderExperimentCard(exp, i)).join('')}</div>`;
  }

  function renderGroupedByCategory(exps) {
    const grouped = {};
    exps.forEach(exp => {
      if (!grouped[exp.category]) grouped[exp.category] = [];
      grouped[exp.category].push(exp);
    });

    return Object.entries(grouped).map(([catKey, catExps]) => {
      const cat = categories[catKey];
      if (!cat) return '';
      return `
        <div class="category-section">
          <div class="category-header">
            <div class="category-icon" style="background: ${cat.color}15; color: ${cat.color};">${cat.icon}</div>
            <h2>${cat.label}</h2>
            <span class="category-count">${catExps.length} experimentos</span>
          </div>
          <div class="experiments-grid">
            ${catExps.map((exp, i) => renderExperimentCard(exp, i)).join('')}
          </div>
        </div>
      `;
    }).join('');
  }

  function renderDetailOverlay(exp) {
    if (!exp) return '';

    const statusLabels = {
      completed: 'Completo',
      partial: 'Parcial',
      blocked: 'Bloqueado',
      external: 'Externo',
    };

    const modelsHtml = exp.models.map(m =>
      `<div class="detail-model-item"><span class="model-dot"></span>${m}</div>`
    ).join('');

    const tagsHtml = exp.techniques.map(t => `<span class="tag highlight">${t}</span>`).join('');

    const metricHtml = exp.metric ? `
      <div class="detail-info-item">
        <div class="info-label">${exp.metric.label}</div>
        <div class="info-value mono">${exp.metric.value}</div>
      </div>
    ` : '';

    return `
      <div class="detail-panel">
        <button class="detail-close" onclick="App.closeDetail()">\u2715</button>
        <div class="detail-header">
          <span class="card-number">#${String(exp.id).padStart(2, '0')}</span>
          <span class="status-badge ${exp.status}" style="margin-left:8px">${statusLabels[exp.status]}</span>
          <h2>${exp.title}</h2>
          <p class="detail-desc">${exp.description}</p>
        </div>
        <div class="detail-section">
          <h4>Informacoes</h4>
          <div class="detail-info-grid">
            <div class="detail-info-item">
              <div class="info-label">Categoria</div>
              <div class="info-value">${exp.categoryLabel}</div>
            </div>
            <div class="detail-info-item">
              <div class="info-label">Dataset</div>
              <div class="info-value">${exp.dataset}</div>
            </div>
            ${metricHtml}
            <div class="detail-info-item">
              <div class="info-label">Status</div>
              <div class="info-value">${statusLabels[exp.status]}</div>
            </div>
          </div>
        </div>
        <div class="detail-section">
          <h4>Tecnicas Utilizadas</h4>
          <div class="card-tags">${tagsHtml}</div>
        </div>
        <div class="detail-section">
          <h4>Modelos</h4>
          <div class="detail-models-list">${modelsHtml}</div>
        </div>
        <div class="detail-section">
          <h4>Script / Caminho</h4>
          <div class="detail-script-path">${exp.script}</div>
        </div>
        <div class="detail-section">
          <h4>Detalhes Adicionais</h4>
          <p style="font-size:0.82rem;color:var(--text-secondary);line-height:1.6;">${exp.details}</p>
        </div>
      </div>
    `;
  }

  /* -------------------------------------------------------
     RENDER MAIN
     ------------------------------------------------------- */
  function render() {
    const sidebar = document.getElementById('sidebar');
    const topbar = document.getElementById('topbar');
    const content = document.getElementById('page-content');

    sidebar.innerHTML = renderSidebar();
    topbar.innerHTML = renderTopbar();

    let html = '';

    if (state.activeCategory === 'all' && state.searchQuery === '' && state.statusFilter === 'all') {
      html += renderOverview();
      html += renderCharts();
    }

    html += renderExperiments();
    content.innerHTML = html;

    // Draw charts after DOM is ready
    requestAnimationFrame(() => {
      drawAllCharts();
      animateMetricBars();
    });
  }

  function drawAllCharts() {
    const catCanvas = document.getElementById('chart-categories');
    const statusCanvas = document.getElementById('chart-status');
    const evoCanvas = document.getElementById('chart-evolution');
    const radarCanvas = document.getElementById('chart-radar');

    if (catCanvas) {
      const catStats = getCategoryStats();
      const data = Object.entries(categories)
        .filter(([k]) => k !== 'all')
        .map(([k, v]) => ({
          label: v.label,
          count: catStats[k] || 0,
          color: v.color,
        }))
        .sort((a, b) => b.count - a.count);
      Charts.drawCategoryChart(catCanvas, data);
    }

    if (statusCanvas) {
      const statusCounts = {};
      experiments.forEach(e => {
        statusCounts[e.status] = (statusCounts[e.status] || 0) + 1;
      });
      Charts.drawStatusChart(statusCanvas, [
        { label: 'Completos', count: statusCounts.completed || 0, color: Charts.COLORS.completed },
        { label: 'Parciais', count: statusCounts.partial || 0, color: Charts.COLORS.partial },
        { label: 'Bloqueados', count: statusCounts.blocked || 0, color: Charts.COLORS.blocked },
        { label: 'Externos', count: statusCounts.external || 0, color: Charts.COLORS.external },
      ]);
    }

    if (evoCanvas) {
      Charts.drawEvolutionChart(evoCanvas, sentipredEvolution);
    }

    if (radarCanvas) {
      const techGroups = [
        { label: 'TF-IDF / NLP', count: 0 },
        { label: 'Ensemble', count: 0 },
        { label: 'Deep Learning', count: 0 },
        { label: 'AutoML', count: 0 },
        { label: 'Time Series', count: 0 },
        { label: 'Computer Vision', count: 0 },
        { label: 'Anomaly/Stats', count: 0 },
        { label: 'RL / Trading', count: 0 },
      ];
      experiments.forEach(exp => {
        const techs = exp.techniques.join(' ').toLowerCase();
        if (techs.includes('tf-idf') || techs.includes('nlp') || techs.includes('ner') || techs.includes('spacy')) techGroups[0].count++;
        if (techs.includes('ensemble') || techs.includes('voting') || techs.includes('stacking') || techs.includes('bagging')) techGroups[1].count++;
        if (techs.includes('transformer') || techs.includes('bert') || techs.includes('cnn') || techs.includes('distilbert') || techs.includes('deep') || techs.includes('yolo')) techGroups[2].count++;
        if (techs.includes('automl') || techs.includes('flaml') || techs.includes('autogluon') || techs.includes('optuna')) techGroups[3].count++;
        if (techs.includes('time series') || techs.includes('prophet') || techs.includes('sarima') || techs.includes('forecast') || techs.includes('lightgbm')) techGroups[4].count++;
        if (techs.includes('image') || techs.includes('face') || techs.includes('object detection') || techs.includes('pytorch') || techs.includes('tensorflow')) techGroups[5].count++;
        if (techs.includes('anomaly') || techs.includes('drift') || techs.includes('z-score') || techs.includes('isolation') || techs.includes('shap') || techs.includes('explainability')) techGroups[6].count++;
        if (techs.includes('reinforcement') || techs.includes('q-learning') || techs.includes('trading')) techGroups[7].count++;
      });
      Charts.drawTechRadar(radarCanvas, techGroups);
    }
  }

  function animateMetricBars() {
    document.querySelectorAll('.metric-bar-fill').forEach(bar => {
      const width = bar.style.width;
      bar.style.width = '0%';
      setTimeout(() => { bar.style.width = width; }, 100);
    });
  }

  /* -------------------------------------------------------
     EVENT HANDLERS
     ------------------------------------------------------- */

  function setCategory(cat) {
    state.activeCategory = cat;
    state.searchQuery = '';
    state.statusFilter = 'all';
    closeSidebar();
    render();
    document.getElementById('page-content').scrollTo({ top: 0, behavior: 'smooth' });
  }

  function setSearch(query) {
    state.searchQuery = query;
    // Debounced render
    clearTimeout(App._searchTimeout);
    App._searchTimeout = setTimeout(render, 200);
  }

  function setStatus(status) {
    state.statusFilter = status;
    render();
  }

  function showDetail(id) {
    const exp = experiments.find(e => e.id === id);
    if (!exp) return;

    const overlay = document.getElementById('detail-overlay');
    overlay.innerHTML = renderDetailOverlay(exp);
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeDetail() {
    const overlay = document.getElementById('detail-overlay');
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  }

  function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    state.sidebarOpen = !state.sidebarOpen;
    sidebar.classList.toggle('open', state.sidebarOpen);
    if (overlay) overlay.style.display = state.sidebarOpen ? 'block' : 'none';
  }

  function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    state.sidebarOpen = false;
    sidebar.classList.remove('open');
    if (overlay) overlay.style.display = 'none';
  }

  /* -------------------------------------------------------
     INIT
     ------------------------------------------------------- */
  function init() {
    render();

    // Close detail on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeDetail();
    });

    // Close detail on overlay click
    document.getElementById('detail-overlay').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeDetail();
    });

    // Resize charts
    let resizeTimeout;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => drawAllCharts(), 250);
    });
  }

  /* -------------------------------------------------------
     Public API
     ------------------------------------------------------- */
  return {
    init,
    render,
    setCategory,
    setSearch,
    setStatus,
    showDetail,
    closeDetail,
    toggleSidebar,
    closeSidebar,
    _searchTimeout: null,
  };

})();

// Boot
document.addEventListener('DOMContentLoaded', App.init);
