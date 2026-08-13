/* ============================================================
   MLOps Experiments Dashboard - Application Logic
   All experiment data + rendering + filters + search + navigation
   ============================================================ */

const App = (() => {

  const GH_BASE = 'https://github.com/pedro-morato/mlops-experiments/blob/main/';

  /* -------------------------------------------------------
     EXPERIMENT DATA — complete inventory of the repository
     ------------------------------------------------------- */
  const experiments = [
    // ── Reinforcement Learning ──────────────────────────────
    { id:200, title:'RL-AutoML (Q-Learning Agent)', category:'rl', categoryLabel:'Reinforcement Learning', status:'completed',
      description:'Agente autonomo de Q-Learning que encontra hiperparametros de LightGBM explorando a equacao de Bellman.',
      techniques:['Q-Learning','LightGBM','AutoML','Epsilon-Greedy','Bellman'],
      metric:{label:'Max F1',value:'97.5%+',percent:97.5},
      script:'experiments/reinforcement_learning/rl_automl_qlearning.ipynb',
      readme:'experiments/reinforcement_learning/README.md',
      models:['Q-Table Agent','LGBMClassifier'], dataset:'Breast Cancer (Sklearn)',
      details:'125 estados (LR, Leaves, Depth). Convergencia quase instantanea apos exploracao.' },
    { id:201, title:'RL-AutoML Senti-Pred (Full Scale)', category:'rl', categoryLabel:'Reinforcement Learning', status:'completed',
      description:'Q-Learning em escala massiva: LinearSVC com 74k linhas e 100k features TF-IDF.',
      techniques:['Q-Learning','LinearSVC','NLP','100k Features'],
      metric:{label:'Stress Test',value:'100k feats',percent:90},
      script:'experiments/reinforcement_learning/rl_sentipred_automl.ipynb',
      readme:'experiments/reinforcement_learning/README.md',
      models:['Q-Table Agent','LinearSVC'], dataset:'Twitter Sentiment (74k)',
      details:'Centenas de fits iterativos em matriz de 100k features.' },
    { id:202, title:'RL-AutoML Sales Forecast (Proxy Big Data)', category:'rl', categoryLabel:'Reinforcement Learning', status:'completed',
      description:'Q-Learning com Proxy Training em 5.6M transacoes. Recompensa inversa (queda de MAE).',
      techniques:['Q-Learning','Proxy Training','LightGBM','Inverse Reward','Big Data'],
      metric:{label:'MAE',value:'1.4297',percent:95},
      script:'experiments/sales-forecast/rl_proxy_sales_full.ipynb',
      readme:'experiments/reinforcement_learning/README.md',
      models:['Q-Table','LGBMRegressor'], dataset:'Retail Sales 5.6M',
      details:'Proxy (n_est=50, bagging=0.15) encontra config otima; MAE 1.4297 vs Optuna 1.4218.' },

    // ── NLP — Sentiment ─────────────────────────────────────
    { id:1, title:'Senti-Pred Pipeline A (Agressivo)', category:'nlp-sentiment', categoryLabel:'NLP - Sentimento', status:'completed',
      description:'Pipeline agressivo: remove hashtags/pontuacao/numeros. TF-IDF 70k + ExtraTrees F1 0.982.',
      techniques:['TF-IDF 70k','ExtraTrees','LinearSVC','Bigrams'],
      metric:{label:'F1',value:'0.982',percent:98.2},
      script:'experiments/nlp/twitter-entity-sentiment/senti-pred_pipeline.ipynb',
      readme:'experiments/nlp/README.md',
      models:['ExtraTrees','LinearSVC','LR','MNB'], dataset:'Twitter Entity Sentiment (74k)',
      details:'4 fases de evolucao. Fase 4: LinearSVC C=10 atinge 0.982.' },
    { id:2, title:'Senti-Pred Pipeline B (Conservador)', category:'nlp-sentiment', categoryLabel:'NLP - Sentimento', status:'completed',
      description:'Pipeline conservador: preserva hashtags/pontuacao. LinearSVC C=19 F1 0.983.',
      techniques:['TF-IDF','LinearSVC','Conservative Cleaning'],
      metric:{label:'F1',value:'0.983',percent:98.3},
      script:'experiments/nlp/twitter-entity-sentiment/twitter-sentiment-analysis.ipynb',
      readme:'experiments/nlp/README.md',
      models:['LinearSVC','ExtraTrees'], dataset:'Twitter Entity Sentiment (74k)',
      details:'Preserva conteudo de hashtags e contrações idiomáticas.' },
    { id:3, title:'Senti-Pred Remake2 (Pipeline C)', category:'nlp-sentiment', categoryLabel:'NLP - Sentimento', status:'completed',
      description:'Voting Ensemble (LinearSVC+LR) com TF-IDF 100k e 4-grams. Recorde 97.80%.',
      techniques:['TF-IDF 100k','4-grams','Voting Ensemble','LinearSVC'],
      metric:{label:'Accuracy',value:'97.80%',percent:97.8},
      script:'experiments/nlp/twitter-entity-sentiment/senti-pred-variations/',
      readme:'experiments/nlp/twitter-entity-sentiment/senti-pred-variations/README.md',
      models:['Voting (LinearSVC+LogReg)'], dataset:'Twitter Sentiment (4 classes)',
      details:'Pipeline C: vetorizacao extrema com lematizacao e expansao de contracoes.' },
    { id:4, title:'Pipeline A vs B vs C (Duelo + Ablacoes)', category:'nlp-sentiment', categoryLabel:'NLP - Sentimento', status:'completed',
      description:'Comparativo rigoroso das 3 pipelines com ablation (n-grams, vocab, limpeza) e teste McNemar.',
      techniques:['Ablation Study','McNemar Test','TF-IDF','LinearSVC'],
      metric:{label:'Best F1',value:'0.9857',percent:98.57},
      script:'experiments/nlp/twitter-entity-sentiment/pipelines_abc_comparison/',
      readme:'experiments/nlp/twitter-entity-sentiment/pipelines_abc_comparison/README.md',
      models:['LinearSVC','ExtraTrees','LogReg'], dataset:'Twitter Sentiment',
      details:'Melhor F1 do estudo: 0.9857 combinando limpeza A + vetorizador C. Diferencas < 1pp nao significativas.' },
    { id:5, title:'Ensemble Pyramid (6 Camadas)', category:'nlp-sentiment', categoryLabel:'NLP - Sentimento', status:'completed',
      description:'Piramide hierarquica com 6 camadas de meta-ensembles (Bagging/Voting/Stacking).',
      techniques:['Bagging','Voting','Stacking','RL Meta-Learner','TF-IDF 70k'],
      metric:{label:'F1',value:'~98%+',percent:98},
      script:'experiments/ensemble_pyramid.ipynb', readme:null,
      models:['LR','LinearSVC','NB','CNB','Ridge','RF','ExtraTrees','Meta-Stacking'], dataset:'Twitter Sentiment (4 classes)',
      details:'6 camadas progressivas. Classes leves PreFittedSoftVoting evitam re-treino.' },
    { id:6, title:'Versatile Ensemble Pyramid (AutoML RL)', category:'nlp-sentiment', categoryLabel:'NLP - Sentimento', status:'completed',
      description:'Motor AutoML com RL Meta-Learner que decide arquitetura da piramide dinamicamente.',
      techniques:['RL Meta-Learner','Thompson Sampling','AutoML CLI','TF-IDF'],
      metric:{label:'AutoML',value:'CLI',percent:95},
      script:'experiments/ensemble_pyramid.py', readme:null,
      models:['RL Agent','Multi-Ensemble'], dataset:'Twitter Sentiment',
      details:'CLI com --layers, --strategy, --epsilon, --jitter. MLflow auto-tracking.' },

    // ── NLP — Classification & Representations ──────────────
    { id:10, title:'Twitter Methods Comparison (5 Paradigms)', category:'nlp-class', categoryLabel:'NLP - Classificacao', status:'completed',
      description:'TF-IDF+LinearSVC vs DistilBERT vs TextCNN vs BiLSTM vs Sentence-BERT no dataset completo.',
      techniques:['TF-IDF+LinearSVC','DistilBERT','TextCNN','BiLSTM','Sentence-BERT','Mamba'],
      metric:{label:'Best Acc',value:'0.980',percent:98},
      script:'experiments/nlp/twitter-entity-sentiment/NLP-twitter-methods-comparasion.ipynb',
      readme:'experiments/nlp/README.md',
      models:['LinearSVC','DistilBERT','TextCNN','BiLSTM','Sentence-BERT'], dataset:'Twitter (74k)',
      details:'TF-IDF+LinearSVC 0.98 em 4.35s. DistilBERT 0.971 em 40min. TextCNN melhor custo-beneficio neural.' },
    { id:11, title:'AG News Classification (Low-Data)', category:'nlp-class', categoryLabel:'NLP - Classificacao', status:'completed',
      description:'DistilBERT fine-tune vs TF-IDF em 1k amostras. Transformer vence em low-data.',
      techniques:['DistilBERT','TF-IDF','Grid Search','Fine-tuning'],
      metric:{label:'Accuracy',value:'0.835',percent:83.5},
      script:'experiments/nlp/ag-news-classification.ipynb',
      readme:'experiments/nlp/README.md',
      models:['DistilBERT','LinearSVC','ExtraTrees'], dataset:'AG News (4 classes, 1k train)',
      details:'DistilBERT 0.835 vs TF-IDF+LinearSVC 0.765. Grid search otimo em 3-4k features.' },
    { id:12, title:'Multi-Task Learning (MMoE)', category:'nlp-class', categoryLabel:'NLP - Classificacao', status:'completed',
      description:'MMoE com go_emotions. TF-IDF 15k + Focal Loss + ExtraTrees supera redes neurais.',
      techniques:['MMoE','Focal Loss','ExtraTrees','TF-IDF 15k','Deep Learning'],
      metric:{label:'F1-weighted',value:'0.9643',percent:96.4},
      script:'experiments/nlp/nlp-multi-task-classification.ipynb',
      readme:'experiments/nlp/README.md',
      models:['MMoE Neural','ExtraTrees','LinearSVC','LightGBM'], dataset:'go_emotions (43k)',
      details:'ExtraTrees 0.9643 bate todas as redes. MMoE+Focal Loss 0.9566 com features esparsas.' },
    { id:13, title:'Logistic Regression Multiclass Strategies', category:'nlp-class', categoryLabel:'NLP - Classificacao', status:'completed',
      description:'Multinomial vs OvR vs OvO com varios solvers e valores de C.',
      techniques:['Logistic Regression','Multinomial','OvR','OvO','TF-IDF'],
      metric:{label:'Best Acc',value:'0.982',percent:98.2},
      script:'experiments/nlp/twitter-entity-sentiment/logistic-regression-multiclass.ipynb',
      readme:'experiments/nlp/README.md',
      models:['Multinomial(lbfgs)','OvR(saga)','OvO(liblinear)'], dataset:'Twitter Sentiment',
      details:'Max diff entre strategies: 0.4pp. Multinomial lbfgs C=10 vence.' },
    { id:14, title:'Feature Engineering NLP', category:'nlp-class', categoryLabel:'NLP - Classificacao', status:'completed',
      description:'Hashing trick supera TF-IDF (0.986 vs 0.977). Word+char n-grams dao +0.5pp.',
      techniques:['Hashing Trick','TF-IDF','Word+Char N-grams','Domain Features'],
      metric:{label:'Best Acc',value:'0.986',percent:98.6},
      script:'experiments/nlp/twitter-entity-sentiment/feature-engineering-nlp.ipynb',
      readme:'experiments/nlp/README.md',
      models:['LinearSVC','ExtraTrees'], dataset:'Twitter Sentiment',
      details:'Hashing trick 262k features sem custo IDF. Trees so ganham com domain knowledge.' },
    { id:15, title:'Mamba SSM (Twitter)', category:'nlp-class', categoryLabel:'NLP - Classificacao', status:'partial',
      description:'State-Space Model (130M params) para classificacao de sentimento. Depende de CUDA.',
      techniques:['Mamba','SSM','State-Space','Deep Learning'],
      metric:null,
      script:'experiments/nlp/twitter-entity-sentiment/run_twitter_mamba.ipynb',
      readme:'experiments/nlp/README.md',
      models:['Mamba SSM (130M)'], dataset:'Twitter Sentiment',
      details:'TBD: overhead de projecoes lineares em textos curtos. CUDA/Triton necessario.' },
    { id:16, title:'NLP Regression — Wine Scores', category:'nlp-class', categoryLabel:'NLP - Classificacao', status:'completed',
      description:'Ridge vs LightGBM prevendo pontuacao de vinhos por texto. Linear vence esparso.',
      techniques:['TF-IDF 15k','Ridge Regression','LightGBM','MLflow'],
      metric:{label:'MAE',value:'1.13',percent:88},
      script:'experiments/nlp-regression-wine/nlp_regression_wine.ipynb',
      readme:'experiments/nlp-regression-wine/README.md',
      models:['Ridge Regressor','LightGBM Regressor'], dataset:'Wine Reviews (Kaggle)',
      details:'Ridge MAE 1.13 R2 0.69 vs LightGBM MAE 1.47 R2 0.63. Linear prospera em espaco esparso.' },

    // ── NLP — Hierarchical ──────────────────────────────────
    { id:40, title:'Hierarchical Classification (20 Newsgroups)', category:'hierarchical', categoryLabel:'Hierarquico', status:'completed',
      description:'Flat vs hierarquico local-por-no com TF-IDF word+char e LinearSVC.',
      techniques:['TF-IDF word+char','LinearSVC','Hierarchical Classifier','20 Newsgroups'],
      metric:{label:'Flat Acc',value:'0.7188',percent:71.9},
      script:'experiments/hierarchical/classificacao_hierarquica.ipynb',
      readme:'experiments/hierarchical/README.md',
      models:['LinearSVC(C=0.15)','LogisticRegression'], dataset:'20 Newsgroups (18k docs, 20 classes)',
      details:'Flat 0.7188 vs Hierarquico 0.6953. HF: 0.7668 vs 0.7516. Pai eh gargalo.' },
    { id:41, title:'Clustering: Flat vs Hierarquico', category:'hierarchical', categoryLabel:'Hierarquico', status:'completed',
      description:'KMeans vs Aglomerativo vs Top-down 2-niveis. Top-down vence no nivel folha.',
      techniques:['KMeans','Agglomerative(Ward)','Top-down Clustering','TF-IDF+SVD'],
      metric:{label:'Top-down NMI',value:'0.360',percent:72},
      script:'experiments/hierarchical/clustering_flat_vs_hierarquico.ipynb',
      readme:'experiments/hierarchical/README.md',
      models:['KMeans','AgglomerativeClustering'], dataset:'20 Newsgroups (3k sample)',
      details:'Top-down Purity 0.398, NMI 0.360 — acima da literatura (0.25-0.45).' },

    // ── Feature Selection (Evolutionary) ────────────────────
    { id:45, title:'Feature Selection Evolucionaria (GAAP/MO-DE)', category:'feature-selection', categoryLabel:'Feature Selection', status:'completed',
      description:'NSGA-II e Differential Evolution multiobjetivo vs SelectKBest/Boruta/RF importance.',
      techniques:['NSGA-II','MO-DE','DEAP','SelectKBest','Boruta','Pareto Front'],
      metric:{label:'R2 (23 feats)',value:'0.694',percent:82},
      script:'experiments/feature_selection_ea/feature_selection_ea.ipynb',
      readme:'experiments/feature_selection_ea/README.md',
      models:['Ridge','LogisticRegression','GAAP','MO-DE'], dataset:'California Housing + Twitter',
      details:'EA vence em features interativas (California). Classicos bastam em bag-of-words.' },

    // ── Computer Vision ─────────────────────────────────────
    { id:50, title:'CV Methods Comparison (CIFAR-10)', category:'cv', categoryLabel:'Computer Vision', status:'completed',
      description:'HOG+SVM vs ResNet18 vs ViT no CIFAR-10. ViT atinge 0.9805.',
      techniques:['HOG+SVM','ResNet18','ViT','Fine-tuning','ImageNet-21k'],
      metric:{label:'ViT Acc',value:'0.9805',percent:98.1},
      script:'experiments/computer_vision/cv-methods-comparison.ipynb',
      readme:'experiments/computer_vision/README.md',
      models:['HOG+SVM','ResNet18','ViT-base-patch16-224'], dataset:'CIFAR-10 (50k/10k)',
      details:'ViT 0.9805 > ResNet18 0.9362 > HOG 0.3970. Manual features falham em baixa resolucao.' },
    { id:51, title:'Animal Multi-Label (4 Abordagens)', category:'cv', categoryLabel:'Computer Vision', status:'completed',
      description:'ResNet18+Aug, VGG16, CLIP zero-shot, EfficientNet-B0 para classificacao multi-label de pets.',
      techniques:['ResNet18','VGG16','CLIP Zero-shot','EfficientNet','BCEWithLogitsLoss'],
      metric:{label:'ResNet F1',value:'1.000',percent:100},
      script:'experiments/computer_vision/animal-classifier.ipynb',
      readme:'experiments/computer_vision/README.md',
      models:['ResNet18+Aug','VGG16','CLIP','EfficientNet-B0'], dataset:'Pet Images (44, 2 classes)',
      details:'ResNet18+Aug perfeito (F1 1.000). CLIP exige calibracao cuidadosa de threshold.' },
    { id:52, title:'Face Recognition App', category:'cv', categoryLabel:'Computer Vision', status:'completed',
      description:'App de reconhecimento facial com 3 modos: LBPH, CNN e Transfer Learning (YuNet+MobileNetV2).',
      techniques:['LBPH','CNN','YuNet','MobileNetV2','Face Detection'],
      metric:null,
      script:'experiments/computer_vision/face_recognition_app.ipynb',
      readme:'experiments/computer_vision/README.md',
      models:['LBPH','CNN','MobileNetV2+YuNet'], dataset:'Local Face Dataset',
      details:'Coleta por upload, treino e predicao. LBPH roda em CPU.' },
    { id:53, title:'YOLO Object Detection', category:'cv', categoryLabel:'Computer Vision', status:'completed',
      description:'Deteccao de objetos via YOLOv3-tiny COCO usando OpenCV DNN.',
      techniques:['YOLOv3-tiny','OpenCV DNN','Object Detection'],
      metric:null,
      script:'experiments/computer_vision/yolo_notebook.ipynb',
      readme:'experiments/computer_vision/README.md',
      models:['YOLOv3-tiny'], dataset:'COCO / Custom',
      details:'Upload de imagem e deteccao one-stage. Suporta modelos custom.' },
    { id:54, title:'Knowledge Distillation — CIFAR-10 (3 Families)', category:'cv', categoryLabel:'Computer Vision', status:'completed',
      description:'Response-based (Logit), Feature-based (FitNets), Relation-based (RKD) e Hybrid KD.',
      techniques:['Logit KD','FitNets','RKD','ResNet18 Teacher','CNN Student'],
      metric:null,
      script:'experiments/computer_vision/kd-cifar10-comparison.ipynb',
      readme:'experiments/computer_vision/README.md',
      models:['ResNet18 (Teacher)','CNN 1.1M (Student)'], dataset:'CIFAR-10 (50k/10k)',
      details:'Teacher ResNet18 fine-tuned 224x224 (~0.91). Student CNN ~10x menor. 5 cenarios comparados.' },

    // ── Recommender Systems ─────────────────────────────────
    { id:55, title:'MovieLens RecSys (8 Paradigms)', category:'recsys', categoryLabel:'Recommender Systems', status:'completed',
      description:'Popularidade, KNN User/Item, SVD, NCF, Two-Tower, LightGBM+FE, BPR no MovieLens 100k.',
      techniques:['SVD','KNN','NCF','Two-Tower','LightGBM','BPR','Matrix Factorization'],
      metric:{label:'Two-Tower RMSE',value:'0.9297',percent:93},
      script:'experiments/recommender_systems/movielens-recsys.ipynb',
      readme:'experiments/recommender_systems/README.md',
      models:['SVD','NCF','Two-Tower','LightGBM','BPR','KNN'], dataset:'MovieLens 100k (93.7% sparsity)',
      details:'Two-Tower 0.9297. Cold-start demo: SVD recomenda Empire Strikes Back para perfil Star Wars.' },
    { id:56, title:'MovieLens AutoRec (10 Models)', category:'recsys', categoryLabel:'Recommender Systems', status:'completed',
      description:'Item-AutoRec vence todos os 10 modelos com RMSE 0.9054. Autoencoder com masked MSE.',
      techniques:['AutoRec','Autoencoder','Masked MSE','Collaborative Filtering'],
      metric:{label:'Item-AutoRec RMSE',value:'0.9054',percent:91},
      script:'experiments/recommender_systems/movielens-autorec.ipynb',
      readme:'experiments/recommender_systems/README.md',
      models:['Item-AutoRec','User-AutoRec','SVD','Two-Tower','NCF'], dataset:'MovieLens 100k',
      details:'MLflow tracked. Item-based >> User-based em alta esparsidade.' },
    { id:57, title:'Image Recommender (Visual Similarity)', category:'recsys', categoryLabel:'Recommender Systems', status:'completed',
      description:'Recomendacao por similaridade visual: embeddings ResNet + cosine similarity.',
      techniques:['ResNet Embeddings','Cosine Similarity','L2 Normalization'],
      metric:null,
      script:'experiments/recommender_systems/image_recommender.ipynb',
      readme:'experiments/recommender_systems/README.md',
      models:['ResNet Feature Extractor'], dataset:'Local Images (~30)',
      details:'Pipeline: collect → embed → L2 normalize → cosine top-K. 30 imgs indexadas em 4.2s.' },

    // ── Time Series & Forecast ──────────────────────────────
    { id:60, title:'Prophet + Optuna (Temperature)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Tuning bayesiano de Prophet com Optuna minimizando MAE em Time Series CV.',
      techniques:['Prophet','Optuna','Bayesian Optimization','Cross Validation'],
      metric:{label:'MAE',value:'1.96',percent:85},
      script:'experiments/time_series/temperature_forecasting_prophet.ipynb',
      readme:'experiments/time_series/README.md',
      models:['Prophet (Optuna)'], dataset:'Daily Min Temperatures',
      details:'Multiplicative seasonality. changepoint_prior_scale e seasonality_prior_scale tunados.' },
    { id:61, title:'Prophet vs LightGBM (Temperature)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'LightGBM com lags+rolling supera Prophet em ruido diario abrupto.',
      techniques:['LightGBM','Prophet','Lag Features','Rolling Windows'],
      metric:{label:'LGBM MAE',value:'1.7344',percent:90},
      script:'experiments/time_series/temperature_forecasting_prophet.ipynb',
      readme:'experiments/time_series/README.md',
      models:['LightGBM','Prophet'], dataset:'Daily Temperatures',
      details:'LightGBM 1.7344 vs Prophet 1.96. Arvores reagem melhor a ruido abrupto.' },
    { id:62, title:'Sales Forecast V2.2 (Hackathon)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'LightGBM com 32 features + Optuna pruning + MLflow + Docker + 10 testes Pytest.',
      techniques:['LightGBM','Optuna','MLflow','Docker','Pytest','32 Features'],
      metric:{label:'MAE',value:'1.4218',percent:95},
      script:'experiments/sales-forecast/Predictive_Sales_Pipeline.ipynb',
      readme:'experiments/sales-forecast/README.md',
      models:['LightGBM Regressor'], dataset:'Sales 2022 (5.6M rows)',
      details:'V2→V2.2: MAE 2.5769→1.4218 (-44.8%). 10 categoricas de alta cardinalidade.' },
    { id:63, title:'AE Embedding Experiments (Sales)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Autoencoder embeddings como features e clustering para sales forecast. Todos pioram.',
      techniques:['Autoencoder','MLP','Causal Mask','K-means Clustering'],
      metric:{label:'Causal AE',value:'-0.14%',percent:50},
      script:'experiments/sales-forecast/ae_embedding_experiments.ipynb',
      readme:'experiments/sales-forecast/README.md',
      models:['MLP Autoencoder (47→8)'], dataset:'Sales 709k series',
      details:'AE causal neutro (-0.14%). Naive tem vazamento (+20%). Clustering piora tudo.' },
    { id:64, title:'Decomposition vs Regression (Sales)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Testa decomposicao (nivel+tendencia+sazonalidade) vs regressao LightGBM.',
      techniques:['Decomposition','LightGBM','Time Series'],
      metric:null,
      script:'experiments/sales-forecast/decomposition_vs_regression.ipynb',
      readme:'experiments/sales-forecast/README.md',
      models:['LightGBM','Decomposition'], dataset:'Sales 5.6M',
      details:'Regressao supervisionada supera decomposicao classica.' },
    { id:65, title:'Knowledge Distillation (Time Series)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'LSTM+Attention→TCN: Student retém 103.9% do Teacher. LGBM→LGBM falha.',
      techniques:['LSTM','TCN','Attention','Knowledge Distillation'],
      metric:{label:'Student-KD',value:'103.9%',percent:99},
      script:'experiments/time_series/knowledge_distillation-time_series.ipynb',
      readme:'experiments/time_series/README.md',
      models:['LSTM+Attn (1.44M)','TCN (228k)'], dataset:'Hourly Electricity',
      details:'KD funciona para redes densas. Falha para arvores (professor overfita).' },
    { id:66, title:'Anomaly Detection (5 Techniques)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Z-Score vence (F1 0.9954, 0 alarmes falsos) em temperatura de Melbourne.',
      techniques:['Z-Score','Prophet Intervals','Isolation Forest','Elliptic Envelope','LOF'],
      metric:{label:'Z-Score F1',value:'0.9954',percent:99.5},
      script:'experiments/time_series/exp4_anomaly_detection.ipynb',
      readme:'experiments/time_series/README.md',
      models:['Z-Score','IsolationForest','Prophet','EE','LOF'], dataset:'Melbourne Temp (3650 days)',
      details:'Z-Score: 108/109 anomalias, 0 falsos. Prophet interval 99.9% tambem excelente.' },
    { id:67, title:'Benchmark 4x4 (Paradigms × Scenarios)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'SARIMA, Prophet, TCN, LightGBM em 4 datasets. SARIMA vence 2/4.',
      techniques:['SARIMA','Prophet','TCN','LightGBM','Diebold-Mariano'],
      metric:null,
      script:'experiments/time_series/benchmark-ts-paradigms.ipynb',
      readme:'experiments/time_series/README.md',
      models:['SARIMA','Prophet','TCN','LightGBM'], dataset:'CO2, Nile, Sunspots, Synthetic',
      details:'SARIMA 1o em CO2/Nilo. TCN 1o em Sunspots. Prophet 1o em Synthetic. DM test p<0.05 em 3/4.' },
    { id:68, title:'TS Classification (6 Paradigms)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'ROCKET domina 3/3 datasets UEA. DTW baseline robusta mas lenta.',
      techniques:['ROCKET','1-NN+DTW','InceptionTime','TSFresh+RF','Transformer','LightGBM+FE'],
      metric:{label:'ROCKET GunPoint',value:'1.000',percent:100},
      script:'experiments/time_series/time-series-classification.ipynb',
      readme:'experiments/time_series/README.md',
      models:['ROCKET','DTW','InceptionTime','Transformer'], dataset:'GunPoint/ArrowHead/ECG5000',
      details:'ROCKET: 1.000/0.953/0.889. DTW: 36min no ECG5000. Transformer colapsa em ArrowHead.' },
    { id:69, title:'TS + NLP (Stock Sentiment Fusion)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Fusao de series temporais com NLP para direcionamento de mercado sintetico.',
      techniques:['LightGBM','TS+NLP Fusion','Sentiment Features'],
      metric:{label:'NLP-only Acc',value:'0.730',percent:73},
      script:'experiments/time_series/stock-sentiment-ts-nlp.ipynb',
      readme:'experiments/time_series/README.md',
      models:['LightGBM (TS/NLP/TS+NLP)'], dataset:'Synthetic GBM + Headlines',
      details:'NLP-only 0.730, TS+NLP F1 0.720, TS-only 0.492. Noticia defasada domina.' },
    { id:70, title:'Forecast → Direction Classification', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Converte forecast em classificacao de direcao (up/down). Logistica vence.',
      techniques:['Logistic Regression','Random Forest','XGBoost','LightGBM'],
      metric:{label:'Logistic Acc',value:'0.958',percent:95.8},
      script:'experiments/time_series/forecast-classification.ipynb',
      readme:'experiments/time_series/README.md',
      models:['Logistic','RF','XGBoost','LightGBM'], dataset:'fato_vendas (daily agg)',
      details:'Logistica Acc 0.958, AUC 0.967. is_weekend, dow, lag_7 = 59% importancia.' },
    { id:71, title:'TS Feature Engineering — 5 Phases', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Jornada: tsfresh vs manual vs LSTM embeddings vs DWT wavelets vs Optuna.',
      techniques:['tsfresh','Manual FE','LSTM Autoencoder','DWT Wavelets','Optuna'],
      metric:{label:'DWT+Manual MAE',value:'54.19',percent:95},
      script:'experiments/ts_fe/',
      readme:'experiments/ts_fe/README.md',
      models:['Random Forest','LSTM AE'], dataset:'Daily Temp + Beijing PM2.5',
      details:'Fase 4: DWT+Manual MAE 54.19 (melhor). tsfresh destroi performance. Optuna paradox.' },
    { id:72, title:'Watsonx Local Equivalent (Forecast)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Prophet+Optuna/SARIMA/ETS em Electric Production. Equivalente open-source do Watsonx.',
      techniques:['Prophet','SARIMA','ETS','Optuna'],
      metric:{label:'MAPE',value:'3.90%',percent:96},
      script:'experiments/time_series/ibm-watsonx-local-timeseries.ipynb',
      readme:'experiments/ibm-experiments/README.md',
      models:['Prophet+Optuna','SARIMA','ETS','Naive'], dataset:'Electric Production',
      details:'Prophet+Optuna RMSE 3.5583. SARIMA empata MAPE 3.90% em 24x mais rapido.' },
    { id:73, title:'Databricks Local Equivalent', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'Prophet+Optuna/SARIMA/ETS em vendas sinteticas. Equivalente open-source do Databricks.',
      techniques:['Prophet','Optuna','SARIMA','ETS'],
      metric:{label:'sMAPE',value:'5.66%',percent:94},
      script:'experiments/time_series/databricks-forecast-local-equivalent.ipynb',
      readme:'experiments/databricks-forecast/README.md',
      models:['Prophet+Optuna','SARIMA','ETS'], dataset:'Synthetic Sales',
      details:'Prophet+Optuna sMAPE 5.66% (+11.4% vs baseline). Substitui Databricks AutoML.' },
    { id:74, title:'Sktime vs Hybrid (Phase 6)', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'sktime WindowSummarizer vs modelo hibrido com Wavelets no Beijing PM2.5.',
      techniques:['sktime','WindowSummarizer','Wavelets','Random Forest'],
      metric:{label:'MAE',value:'52.79',percent:99},
      script:'experiments/time_series/sktime_vs_hybrid_ts.ipynb',
      readme:'experiments/ts_fe/README.md',
      models:['Random Forest'], dataset:'Beijing PM2.5',
      details:'sktime WindowSummarizer quebra recorde com MAE 52.79 em menos de 1s.' },

    // ── Anomaly Detection (Standalone) ──────────────────────
    { id:80, title:'Anomaly: Supervised vs Unsupervised', category:'anomaly', categoryLabel:'Deteccao de Anomalias', status:'completed',
      description:'Random Forest vs Isolation Forest no NAB machine temperature.',
      techniques:['Random Forest','Isolation Forest','Feature Engineering'],
      metric:null,
      script:'experiments/anomaly_detection_comparison.ipynb', readme:null,
      models:['RandomForestClassifier','IsolationForest'], dataset:'NAB Machine Temp',
      details:'Comparacao supervisionado vs nao supervisionado para deteccao de anomalias.' },
    { id:81, title:'Anomaly: 4 Paradigms Enhanced', category:'anomaly', categoryLabel:'Deteccao de Anomalias', status:'completed',
      description:'Density, Clustering, Representation Learning (Autoencoder), Binary Classification.',
      techniques:['Isolation Forest','LOF','OCSVM','GMM','KMeans','DBSCAN','Autoencoder','XGBoost','SMOTE'],
      metric:null,
      script:'experiments/anomaly_detection_enhanced.ipynb', readme:null,
      models:['IF','LOF','OCSVM','EE','GMM','KMeans','DBSCAN','AE','RF+SMOTE','XGBoost'], dataset:'NAB Machine Temp (22.7k)',
      details:'4 familias: density, clustering, representation learning, classificacao binaria.' },

    // ── Clustering (Standalone) ─────────────────────────────
    { id:85, title:'Clustering: Unsupervised vs Semi vs Supervised', category:'clustering', categoryLabel:'Clustering', status:'completed',
      description:'KMeans, DBSCAN, Agglomerative, GMM vs RF/KNN em Iris, Blobs, Moons, Circles.',
      techniques:['KMeans','DBSCAN','Agglomerative','GMM','RF','KNN'],
      metric:null,
      script:'experiments/run_clustering_comparison.ipynb', readme:null,
      models:['KMeans','DBSCAN','AgglomerativeClustering','GMM'], dataset:'Iris/Blobs/Moons/Circles',
      details:'Compara ARI/NMI entre abordagens nao supervisionadas e supervisionadas.' },
    { id:86, title:'Supervised Clustering (Concept)', category:'clustering', categoryLabel:'Clustering', status:'completed',
      description:'Conceito de supervised clustering: clustering guiado por labels, avaliado por ARI/NMI.',
      techniques:['KMeans+LDA','GMM+Labels','PCA','RF-guided'],
      metric:null,
      script:'experiments/run_supervised_clustering.ipynb', readme:null,
      models:['KMeans','LDA','GMM'], dataset:'Iris',
      details:'Cluster_id nao tem significado; o que importa eh o agrupamento.' },
    { id:87, title:'Senti-Pred as Supervised Clustering', category:'clustering', categoryLabel:'Clustering', status:'completed',
      description:'Aplica supervised clustering ao dataset Twitter Sentiment com TF-IDF.',
      techniques:['TF-IDF','PCA','TruncatedSVD','LDA','KMeans','DBSCAN'],
      metric:null,
      script:'experiments/run_senti_supervised_clustering.ipynb', readme:null,
      models:['KMeans','LDA','DBSCAN','RF'], dataset:'Twitter Sentiment',
      details:'Reducao dimensional (PCA/SVD/LDA) + clustering em dados textuais.' },

    // ── Forecast Comparison (Standalone) ────────────────────
    { id:88, title:'Forecast: Supervised vs Statistical', category:'timeseries', categoryLabel:'Series Temporais', status:'completed',
      description:'RF, SARIMAX, Prophet, ExpSmoothing, XGBoost em Air Passengers e Sunspots.',
      techniques:['Random Forest','SARIMAX','Prophet','ExponentialSmoothing','XGBoost'],
      metric:null,
      script:'experiments/run_forecast_comparison.ipynb', readme:null,
      models:['RF','SARIMAX','Prophet','ExpSmoothing','XGBoost'], dataset:'Air Passengers + Sunspots',
      details:'Compara abordagens supervisionadas (lag features) vs estatisticas.' },

    // ── Tabular Regression & AutoML ─────────────────────────
    { id:90, title:'Feature Engineering Tabular (10 Techniques)', category:'regression', categoryLabel:'Regressao Tabular', status:'completed',
      description:'10 tecnicas de FE em California Housing com LR, LightGBM, RF. FE assimetrica por modelo.',
      techniques:['Polynomial','PCA','Geo Features','Log Transform','Binning','Standardization'],
      metric:{label:'Best R2',value:'0.8418',percent:84.2},
      script:'experiments/tabular_regression/feature-engineering-tabular.ipynb',
      readme:'experiments/tabular_regression/README.md',
      models:['LinearRegression','LightGBM','RandomForest'], dataset:'California Housing (20.6k)',
      details:'Combined +13.5pp para LR. Geo +0.6pp para LGBM. PCA -9 a -18pp em todos.' },
    { id:91, title:'Price Prediction v1→v3', category:'regression', categoryLabel:'Regressao Tabular', status:'completed',
      description:'Evolucao de pipeline de preco de automoveis ate R2 0.9489 (Random Forest).',
      techniques:['Random Forest','GridSearchCV','log1p Target','One-Hot Encoding'],
      metric:{label:'R2',value:'0.9489',percent:94.9},
      script:'experiments/tabular_regression/price-prediction-multiple-linear-regression.ipynb',
      readme:'experiments/tabular_regression/README.md',
      models:['RandomForest','XGBoost','ElasticNet','Ridge'], dataset:'Car Prices (205 samples)',
      details:'v2: R2 0.8517→0.9489. v3 plateau (dataset limitante). Residuos normais.' },
    { id:92, title:'Watsonx Local AutoML Equivalent', category:'regression', categoryLabel:'Regressao Tabular', status:'completed',
      description:'FLAML, TPOT, 9 baselines em California Housing. XGBoost vence AutoML.',
      techniques:['FLAML','TPOT','XGBoost','AutoML','Ridge','Lasso','SVR'],
      metric:{label:'XGB R2',value:'0.8401',percent:84},
      script:'experiments/tabular_regression/ibm-watsonx-local-automl.ipynb',
      readme:'experiments/ibm-experiments/README.md',
      models:['XGBoost','FLAML(CatBoost)','TPOT','ExtraTrees'], dataset:'California Housing',
      details:'XGBoost manual supera FLAML/TPOT por margem pequena. AutoML = boa baseline.' },

    // ── IBM Watsonx / Databricks (Cloud) ────────────────────
    { id:95, title:'IBM Watsonx — Boston Housing (Cloud)', category:'ibm', categoryLabel:'IBM Watsonx / Databricks', status:'external',
      description:'Notebook original Watsonx AutoAI para regressao. Depende de credenciais cloud.',
      techniques:['Snap ML','AutoAI','IBM Watson'],
      metric:null,
      script:'experiments/ibm-experiments/Boston Housing Price Prediction.ipynb',
      readme:'experiments/ibm-experiments/README.md',
      models:['Snap ML Regressor'], dataset:'Boston Housing',
      details:'Original cloud. Equivalente local: tabular_regression/ibm-watsonx-local-automl.ipynb.' },
    { id:96, title:'IBM Watsonx — Electric Production (Cloud)', category:'ibm', categoryLabel:'IBM Watsonx / Databricks', status:'external',
      description:'Notebook original autoai-ts-libs para forecast. Depende de credenciais cloud.',
      techniques:['autoai-ts-libs','IBM Watson','Time Series'],
      metric:null,
      script:'experiments/ibm-experiments/Electric_Production.ipynb',
      readme:'experiments/ibm-experiments/README.md',
      models:['Snap ML Forecaster'], dataset:'Electric Production',
      details:'Original cloud. Equivalente local: time_series/ibm-watsonx-local-timeseries.ipynb.' },
    { id:97, title:'IBM Watsonx — Sentimentos (Cloud)', category:'ibm', categoryLabel:'IBM Watsonx / Databricks', status:'external',
      description:'Analise de sentimentos original Watsonx. Metricas TBD.',
      techniques:['Snap ML','Sentiment Analysis','NLP'],
      metric:null,
      script:'experiments/ibm-experiments/analise_de_sentimentos.ipynb',
      readme:'experiments/ibm-experiments/README.md',
      models:['Snap ML Classifier'], dataset:'Sentiment Dataset',
      details:'Original cloud. Metricas dependem de execucao no Watsonx.' },
    { id:98, title:'Databricks — Prophet/DeepAR (Cloud)', category:'ibm', categoryLabel:'IBM Watsonx / Databricks', status:'external',
      description:'Notebooks auto-gerados pelo Databricks AutoML. Prophet e DeepAR.',
      techniques:['Prophet','DeepAR','Hyperopt','SparkTrials','Databricks'],
      metric:null,
      script:'experiments/databricks-forecast/',
      readme:'experiments/databricks-forecast/README.md',
      models:['Prophet','DeepAR'], dataset:'quantity_sales_transactions',
      details:'TBD. Equivalente local: time_series/databricks-forecast-local-equivalent.ipynb.' },

    // ── MLOps Production ────────────────────────────────────
    { id:100, title:'MLOps: FastAPI Serving + MLflow Registry', category:'mlops', categoryLabel:'MLOps Producao', status:'completed',
      description:'Pipeline de producao completo: serving, drift PSI, retrain automatico, dashboard vivo.',
      techniques:['FastAPI','MLflow Registry','PSI Drift','Auto-Retrain','Precomputed Forecast'],
      metric:{label:'Warm latency',value:'~90ms',percent:99},
      script:'mlops/serve.py',
      readme:'mlops/README_mlops.md',
      models:['SalesForecasterV2 (LightGBM)'], dataset:'Sales Forecast V2.2',
      details:'Precompute 12 semanas → lookup numpy ~80-100ms. 1.6k× speedup. Cooldown 1800s.' },
  ];

  /* -------------------------------------------------------
     Category metadata
     ------------------------------------------------------- */
  const categories = {
    'all': { label: 'Todos', icon: '\u{1F3AF}', color: '#818cf8' },
    'rl': { label: 'Reinforcement Learning', icon: '\u{1F9BE}', color: '#ef4444' },
    'nlp-sentiment': { label: 'NLP - Sentimento', icon: '\u{1F4AC}', color: '#f472b6' },
    'nlp-class': { label: 'NLP - Classificacao', icon: '\u{1F4F0}', color: '#fb923c' },
    'hierarchical': { label: 'Hierarquico', icon: '\u{1F333}', color: '#c084fc' },
    'feature-selection': { label: 'Feature Selection', icon: '\u{1F9EC}', color: '#e879f9' },
    'cv': { label: 'Computer Vision', icon: '\u{1F441}', color: '#38bdf8' },
    'recsys': { label: 'Recommender Systems', icon: '\u{1F3AC}', color: '#22d3ee' },
    'timeseries': { label: 'Series Temporais', icon: '\u{1F4C8}', color: '#34d399' },
    'anomaly': { label: 'Anomalias', icon: '\u{1F50D}', color: '#a78bfa' },
    'clustering': { label: 'Clustering', icon: '\u{1F52E}', color: '#c4b5fd' },
    'regression': { label: 'Regressao Tabular', icon: '\u{1F4CA}', color: '#f87171' },
    'ibm': { label: 'IBM Watsonx / Databricks', icon: '\u2601\uFE0F', color: '#fbbf24' },
    'mlops': { label: 'MLOps Producao', icon: '\u{1F680}', color: '#4ade80' },
  };

  /* -------------------------------------------------------
     Senti-Pred Evolution data (for timeline chart)
     ------------------------------------------------------- */
  const sentipredEvolution = [
    { label: 'Pipeline A', value: 98.2 },
    { label: 'Pipeline B', value: 98.3 },
    { label: 'Pipeline C', value: 97.8 },
    { label: 'A+B+C Best', value: 98.57, best: true },
    { label: 'Pyramid', value: 98 },
  ];

  /* -------------------------------------------------------
     State
     ------------------------------------------------------- */
  let state = { activeCategory: 'all', searchQuery: '', statusFilter: 'all', sidebarOpen: false };

  /* -------------------------------------------------------
     Computed
     ------------------------------------------------------- */
  function getFilteredExperiments() {
    return experiments.filter(exp => {
      const cat = state.activeCategory === 'all' || exp.category === state.activeCategory;
      const sta = state.statusFilter === 'all' || exp.status === state.statusFilter;
      const q = state.searchQuery.toLowerCase();
      const src = !q || exp.title.toLowerCase().includes(q) || exp.description.toLowerCase().includes(q) ||
        exp.techniques.some(t => t.toLowerCase().includes(q)) || exp.categoryLabel.toLowerCase().includes(q);
      return cat && sta && src;
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
    const c = {};
    experiments.forEach(e => { c[e.category] = (c[e.category] || 0) + 1; });
    return c;
  }

  /* -------------------------------------------------------
     RENDER FUNCTIONS
     ------------------------------------------------------- */

  function renderSidebar() {
    const catStats = getCategoryStats();
    const stats = getStats();
    const navItems = Object.entries(categories).map(([key, cat]) => {
      const count = key === 'all' ? experiments.length : (catStats[key] || 0);
      if (key !== 'all' && count === 0) return '';
      const activeClass = state.activeCategory === key ? 'active' : '';
      return `<div class="sidebar-nav-item ${activeClass}" data-category="${key}" onclick="App.setCategory('${key}')">
        <span class="icon">${cat.icon}</span><span>${cat.label}</span><span class="count">${count}</span>
      </div>`;
    }).join('');
    return `
      <div class="sidebar-header">
        <div class="sidebar-logo">
          <div class="sidebar-logo-icon">\u{1F9EA}</div>
          <div class="sidebar-logo-text"><h2>MLOps Lab</h2><span>Experiments Hub</span></div>
        </div>
      </div>
      <nav class="sidebar-nav">
        <div class="sidebar-section-title">Categorias</div>${navItems}
      </nav>
      <div class="sidebar-stats">
        <div class="sidebar-stats-grid">
          <div class="sidebar-stat"><div class="value">${stats.total}</div><div class="label">Experimentos</div></div>
          <div class="sidebar-stat"><div class="value">${stats.completed}</div><div class="label">Completos</div></div>
          <div class="sidebar-stat"><div class="value">${stats.categoriesCount}</div><div class="label">Categorias</div></div>
          <div class="sidebar-stat"><div class="value">${stats.techniques}</div><div class="label">Tecnicas</div></div>
        </div>
      </div>`;
  }

  function renderTopbar() {
    const catLabel = categories[state.activeCategory]?.label || 'Todos';
    const filtered = getFilteredExperiments();
    return `
      <button class="mobile-menu-btn" onclick="App.toggleSidebar()">\u2630</button>
      <div class="topbar-title">${catLabel} <span>${filtered.length} experimentos</span></div>
      <div class="search-container">
        <span class="search-icon">\u{1F50E}</span>
        <input type="text" class="search-input" placeholder="Buscar experimentos, tecnicas..."
          value="${state.searchQuery}" oninput="App.setSearch(this.value)" />
      </div>
      <div class="topbar-filters">
        <button class="filter-btn ${state.statusFilter==='all'?'active':''}" onclick="App.setStatus('all')">Todos</button>
        <button class="filter-btn ${state.statusFilter==='completed'?'active':''}" onclick="App.setStatus('completed')">Completos</button>
        <button class="filter-btn ${state.statusFilter==='partial'?'active':''}" onclick="App.setStatus('partial')">Parciais</button>
        <button class="filter-btn ${state.statusFilter==='external'?'active':''}" onclick="App.setStatus('external')">Externos</button>
      </div>`;
  }

  function renderOverview() {
    const s = getStats();
    const pct = Math.round((s.completed / s.total) * 100);
    return `
      <div class="overview-section">
        <div class="overview-header">
          <div>
            <h1>Repositorio de <span class="gradient-text">Experimentos MLOps</span></h1>
            <p>Jornada de aprendizado em ML, NLP, CV, Series Temporais, RecSys, MLOps e mais</p>
          </div>
        </div>
        <div class="stats-row">
          <div class="stat-card animate-in animate-in-1">
            <div class="stat-icon" style="background:rgba(129,140,248,0.15);color:#818cf8;">\u{1F9EA}</div>
            <div class="stat-value">${s.total}</div><div class="stat-label">Experimentos</div>
            <div class="stat-change positive">${pct}% completos</div>
          </div>
          <div class="stat-card animate-in animate-in-2">
            <div class="stat-icon" style="background:rgba(52,211,153,0.15);color:#34d399;">\u2714</div>
            <div class="stat-value">${s.completed}</div><div class="stat-label">Completos</div>
          </div>
          <div class="stat-card animate-in animate-in-3">
            <div class="stat-icon" style="background:rgba(244,114,182,0.15);color:#f472b6;">\u{1F4E6}</div>
            <div class="stat-value">${s.categoriesCount}</div><div class="stat-label">Categorias</div>
          </div>
          <div class="stat-card animate-in animate-in-4">
            <div class="stat-icon" style="background:rgba(251,191,36,0.15);color:#fbbf24;">\u{1F527}</div>
            <div class="stat-value">${s.techniques}</div><div class="stat-label">Tecnicas Unicas</div>
          </div>
          <div class="stat-card animate-in animate-in-5">
            <div class="stat-icon" style="background:rgba(167,139,250,0.15);color:#a78bfa;">\u{1F3C6}</div>
            <div class="stat-value">98.57%</div><div class="stat-label">Melhor F1 (NLP)</div>
          </div>
        </div>
      </div>`;
  }

  function renderCharts() {
    return `
      <div class="charts-section">
        <div class="chart-card animate-in animate-in-1">
          <h3>Distribuicao por Categoria</h3><p class="chart-subtitle">Numero de experimentos por area</p>
          <div class="chart-canvas-wrap"><canvas id="chart-categories"></canvas></div>
        </div>
        <div class="chart-card animate-in animate-in-2">
          <h3>Status dos Experimentos</h3><p class="chart-subtitle">Progresso geral</p>
          <div class="chart-canvas-wrap"><canvas id="chart-status"></canvas></div>
        </div>
        <div class="chart-card animate-in animate-in-3">
          <h3>Evolucao Senti-Pred</h3><p class="chart-subtitle">Accuracy dos pipelines</p>
          <div class="chart-canvas-wrap"><canvas id="chart-evolution"></canvas></div>
        </div>
        <div class="chart-card animate-in animate-in-4">
          <h3>Radar de Tecnicas</h3><p class="chart-subtitle">Distribuicao por tipo de abordagem</p>
          <div class="chart-canvas-wrap"><canvas id="chart-radar"></canvas></div>
        </div>
      </div>`;
  }

  function renderExperimentCard(exp, index) {
    const sl = { completed:'Completo', partial:'Parcial', blocked:'Bloqueado', external:'Externo' };
    const mh = exp.metric ? `<div class="card-metric"><span class="metric-label">${exp.metric.label}</span>
      <div class="metric-bar"><div class="metric-bar-fill" style="width:${exp.metric.percent}%"></div></div>
      <span class="metric-value">${exp.metric.value}</span></div>` : '';
    const tags = exp.techniques.slice(0,4).map((t,i) =>
      `<span class="tag ${i===0?'highlight':''}">${t}</span>`).join('');
    return `
      <div class="experiment-card animate-in animate-in-${index%6+1}" data-category="${exp.category}" onclick="App.showDetail(${exp.id})">
        <div class="card-top">
          <span class="card-number">#${String(exp.id).padStart(3,'0')}</span>
          <span class="status-badge ${exp.status}">${sl[exp.status]||exp.status}</span>
        </div>
        <h3 class="card-title">${exp.title}</h3>
        <p class="card-description">${exp.description}</p>
        <div class="card-tags">${tags}</div>${mh}
      </div>`;
  }

  function renderExperiments() {
    const f = getFilteredExperiments();
    if (!f.length) return `<div class="no-results"><div class="no-results-icon">\u{1F50D}</div>
      <h3>Nenhum experimento encontrado</h3><p>Tente ajustar os filtros</p></div>`;
    if (state.activeCategory === 'all' && !state.searchQuery && state.statusFilter === 'all') return renderGrouped(f);
    return `<div class="experiments-grid">${f.map((e,i) => renderExperimentCard(e,i)).join('')}</div>`;
  }

  function renderGrouped(exps) {
    const grouped = {};
    exps.forEach(e => { if (!grouped[e.category]) grouped[e.category] = []; grouped[e.category].push(e); });
    return Object.entries(grouped).map(([ck, ce]) => {
      const c = categories[ck]; if (!c) return '';
      return `<div class="category-section">
        <div class="category-header">
          <div class="category-icon" style="background:${c.color}15;color:${c.color};">${c.icon}</div>
          <h2>${c.label}</h2><span class="category-count">${ce.length} experimentos</span>
        </div>
        <div class="experiments-grid">${ce.map((e,i) => renderExperimentCard(e,i)).join('')}</div>
      </div>`;
    }).join('');
  }

  function renderDetailOverlay(exp) {
    if (!exp) return '';
    const sl = { completed:'Completo', partial:'Parcial', blocked:'Bloqueado', external:'Externo' };
    const models = exp.models.map(m => `<div class="detail-model-item"><span class="model-dot"></span>${m}</div>`).join('');
    const tags = exp.techniques.map(t => `<span class="tag highlight">${t}</span>`).join('');
    const mh = exp.metric ? `<div class="detail-info-item"><div class="info-label">${exp.metric.label}</div>
      <div class="info-value mono">${exp.metric.value}</div></div>` : '';
    const readmeBtn = exp.readme
      ? `<a class="detail-readme-link" href="${GH_BASE}${exp.readme}" target="_blank" rel="noopener">
           \u{1F4D6} Ver README no GitHub \u2197</a>`
      : '';
    return `
      <div class="detail-panel">
        <button class="detail-close" onclick="App.closeDetail()">\u2715</button>
        <div class="detail-header">
          <span class="card-number">#${String(exp.id).padStart(3,'0')}</span>
          <span class="status-badge ${exp.status}" style="margin-left:8px">${sl[exp.status]||exp.status}</span>
          <h2>${exp.title}</h2>
          <p class="detail-desc">${exp.description}</p>
          ${readmeBtn}
        </div>
        <div class="detail-section"><h4>Informacoes</h4>
          <div class="detail-info-grid">
            <div class="detail-info-item"><div class="info-label">Categoria</div><div class="info-value">${exp.categoryLabel}</div></div>
            <div class="detail-info-item"><div class="info-label">Dataset</div><div class="info-value">${exp.dataset}</div></div>
            ${mh}
            <div class="detail-info-item"><div class="info-label">Status</div><div class="info-value">${sl[exp.status]||exp.status}</div></div>
          </div>
        </div>
        <div class="detail-section"><h4>Tecnicas</h4><div class="card-tags">${tags}</div></div>
        <div class="detail-section"><h4>Modelos</h4><div class="detail-models-list">${models}</div></div>
        <div class="detail-section"><h4>Script / Caminho</h4><div class="detail-script-path">${exp.script}</div></div>
        <div class="detail-section"><h4>Detalhes</h4>
          <p style="font-size:0.82rem;color:var(--text-secondary);line-height:1.6;">${exp.details}</p>
        </div>
      </div>`;
  }

  /* -------------------------------------------------------
     RENDER MAIN
     ------------------------------------------------------- */
  function render() {
    document.getElementById('sidebar').innerHTML = renderSidebar();
    document.getElementById('topbar').innerHTML = renderTopbar();
    let html = '';
    if (state.activeCategory === 'all' && !state.searchQuery && state.statusFilter === 'all') {
      html += renderOverview() + renderCharts();
    }
    html += renderExperiments();
    document.getElementById('page-content').innerHTML = html;
    requestAnimationFrame(() => { drawAllCharts(); animateMetricBars(); });
  }

  function drawAllCharts() {
    const catC = document.getElementById('chart-categories');
    const staC = document.getElementById('chart-status');
    const evoC = document.getElementById('chart-evolution');
    const radC = document.getElementById('chart-radar');
    if (catC) {
      const cs = getCategoryStats();
      Charts.drawCategoryChart(catC, Object.entries(categories).filter(([k])=>k!=='all').map(([k,v])=>({
        label:v.label, count:cs[k]||0, color:v.color })).sort((a,b)=>b.count-a.count));
    }
    if (staC) {
      const sc = {};
      experiments.forEach(e => { sc[e.status] = (sc[e.status]||0)+1; });
      Charts.drawStatusChart(staC, [
        {label:'Completos',count:sc.completed||0,color:Charts.COLORS.completed},
        {label:'Parciais',count:sc.partial||0,color:Charts.COLORS.partial},
        {label:'Externos',count:sc.external||0,color:Charts.COLORS.external},
      ]);
    }
    if (evoC) Charts.drawEvolutionChart(evoC, sentipredEvolution);
    if (radC) {
      const tg = [
        {label:'TF-IDF/NLP',count:0},{label:'Ensemble',count:0},{label:'Deep Learning',count:0},
        {label:'AutoML',count:0},{label:'Time Series',count:0},{label:'Computer Vision',count:0},
        {label:'Anomaly/Stats',count:0},{label:'RL/Trading',count:0}
      ];
      experiments.forEach(e => {
        const t = e.techniques.join(' ').toLowerCase();
        if (t.includes('tf-idf')||t.includes('nlp')||t.includes('ner')) tg[0].count++;
        if (t.includes('ensemble')||t.includes('voting')||t.includes('stacking')||t.includes('bagging')) tg[1].count++;
        if (t.includes('transformer')||t.includes('bert')||t.includes('cnn')||t.includes('autoencoder')||t.includes('yolo')) tg[2].count++;
        if (t.includes('automl')||t.includes('flaml')||t.includes('optuna')||t.includes('tpot')) tg[3].count++;
        if (t.includes('time series')||t.includes('prophet')||t.includes('sarima')||t.includes('forecast')||t.includes('lightgbm')) tg[4].count++;
        if (t.includes('image')||t.includes('face')||t.includes('object detection')||t.includes('pytorch')) tg[5].count++;
        if (t.includes('anomaly')||t.includes('drift')||t.includes('z-score')||t.includes('isolation')||t.includes('shap')) tg[6].count++;
        if (t.includes('reinforcement')||t.includes('q-learning')||t.includes('trading')) tg[7].count++;
      });
      Charts.drawTechRadar(radC, tg);
    }
  }

  function animateMetricBars() {
    document.querySelectorAll('.metric-bar-fill').forEach(b => {
      const w = b.style.width; b.style.width = '0%';
      setTimeout(() => { b.style.width = w; }, 100);
    });
  }

  /* -------------------------------------------------------
     EVENT HANDLERS
     ------------------------------------------------------- */
  function setCategory(cat) {
    state.activeCategory = cat; state.searchQuery = ''; state.statusFilter = 'all';
    closeSidebar(); render();
    document.getElementById('page-content').scrollTo({ top: 0, behavior: 'smooth' });
  }
  function setSearch(q) {
    state.searchQuery = q;
    clearTimeout(App._st); App._st = setTimeout(render, 200);
  }
  function setStatus(s) { state.statusFilter = s; render(); }
  function showDetail(id) {
    const exp = experiments.find(e => e.id === id); if (!exp) return;
    const ov = document.getElementById('detail-overlay');
    ov.innerHTML = renderDetailOverlay(exp); ov.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
  function closeDetail() {
    document.getElementById('detail-overlay').classList.remove('active');
    document.body.style.overflow = '';
  }
  function toggleSidebar() {
    const sb = document.getElementById('sidebar'), ov = document.getElementById('sidebar-overlay');
    state.sidebarOpen = !state.sidebarOpen; sb.classList.toggle('open', state.sidebarOpen);
    if (ov) ov.style.display = state.sidebarOpen ? 'block' : 'none';
  }
  function closeSidebar() {
    const sb = document.getElementById('sidebar'), ov = document.getElementById('sidebar-overlay');
    state.sidebarOpen = false; sb.classList.remove('open');
    if (ov) ov.style.display = 'none';
  }

  /* -------------------------------------------------------
     INIT
     ------------------------------------------------------- */
  function init() {
    render();
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDetail(); });
    document.getElementById('detail-overlay').addEventListener('click', e => { if (e.target === e.currentTarget) closeDetail(); });
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawAllCharts, 250); });
  }

  return { init, render, setCategory, setSearch, setStatus, showDetail, closeDetail, toggleSidebar, closeSidebar, _searchTimeout: null };
})();

document.addEventListener('DOMContentLoaded', App.init);
