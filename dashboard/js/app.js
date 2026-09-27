/* ============================================================
   MLOps Experiments Dashboard - Application Logic
   All experiment data + rendering + filters + search + navigation
   ============================================================ */

const App = (() => {

  const GH_BASE = 'https://github.com/PedroM2626/mlops-experiments/blob/main/';

  /* -------------------------------------------------------
     EXPERIMENT DATA — complete inventory of the repository
     ------------------------------------------------------- */
  const experiments = [
    {
      id: 200,
      title: "RL-AutoML (Q-Learning Agent)",
      category: "rl",
      categoryLabel: "Reinforcement Learning",
      status: "completed",
      description: "Autonomous Q-Learning agent that finds LightGBM hyperparameters by exploring the Bellman equation.",
      techniques: ["Q-Learning", "LightGBM", "AutoML", "Epsilon-Greedy", "Bellman"],
      metric: {"label": "Max F1", "value": "97.5%+", "percent": 97.5},
      script: "experiments/reinforcement_learning/rl_automl_qlearning.ipynb",
      readme: "experiments/reinforcement_learning/README.md",
      models: ["Q-Table Agent", "LGBMClassifier"],
      dataset: "Breast Cancer (Sklearn)",
      details: "125 states (LR, Leaves, Depth). Near-instant convergence after exploration."
    },
    {
      id: 201,
      title: "RL-AutoML Senti-Pred (Full Scale)",
      category: "rl",
      categoryLabel: "Reinforcement Learning",
      status: "completed",
      description: "Q-Learning at massive scale: LinearSVC with 74k rows and 100k TF-IDF features.",
      techniques: ["Q-Learning", "LinearSVC", "NLP", "100k Features"],
      metric: {"label": "Best Acc", "value": "98.60%", "percent": 98.6},
      script: "experiments/reinforcement_learning/rl_sentipred_automl.ipynb",
      readme: "experiments/reinforcement_learning/README.md",
      models: ["Q-Table Agent", "LinearSVC"],
      dataset: "Twitter Sentiment (74k)",
      details: "Hundreds of iterative fits on a 100k-feature matrix. RL lifts the 98.00% baseline to 98.60%."
    },
    {
      id: 202,
      title: "RL-AutoML Sales Forecast (Proxy Big Data)",
      category: "rl",
      categoryLabel: "Reinforcement Learning",
      status: "completed",
      description: "Q-Learning with Proxy Training on 5.6M transactions. Inverse reward (MAE reduction).",
      techniques: ["Q-Learning", "Proxy Training", "LightGBM", "Inverse Reward", "Big Data"],
      metric: {"label": "MAE", "value": "1.4297", "percent": 95},
      script: "experiments/sales-forecast/rl_proxy_sales_full.ipynb",
      readme: "experiments/reinforcement_learning/README.md",
      models: ["Q-Table", "LGBMRegressor"],
      dataset: "Retail Sales 5.6M",
      details: "Proxy (n_est=50, bagging=0.15) finds the optimal config; MAE 1.4297 ties Optuna (1.4218) in a fraction of the time."
    },
    {
      id: 1,
      title: "Senti-Pred Pipeline A (Aggressive)",
      category: "nlp-sentiment",
      categoryLabel: "NLP - Sentiment",
      status: "completed",
      description: "Aggressive pipeline: removes hashtags/punctuation/numbers. TF-IDF 70k + ExtraTrees F1 0.982.",
      techniques: ["TF-IDF 70k", "ExtraTrees", "LinearSVC", "Bigrams"],
      metric: {"label": "F1", "value": "0.982", "percent": 98.2},
      script: "experiments/nlp/twitter-entity-sentiment/senti-pred_pipeline.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["ExtraTrees", "LinearSVC", "LR", "MNB"],
      dataset: "Twitter Entity Sentiment (74k)",
      details: "4 evolution phases. Phase 4: LinearSVC C=10 reaches 0.982."
    },
    {
      id: 2,
      title: "Senti-Pred Pipeline B (Conservative)",
      category: "nlp-sentiment",
      categoryLabel: "NLP - Sentiment",
      status: "completed",
      description: "Conservative pipeline: preserves hashtags/punctuation. LinearSVC C=19 F1 0.983.",
      techniques: ["TF-IDF", "LinearSVC", "Conservative Cleaning"],
      metric: {"label": "F1", "value": "0.983", "percent": 98.3},
      script: "experiments/nlp/twitter-entity-sentiment/twitter-sentiment-analysis.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["LinearSVC", "ExtraTrees"],
      dataset: "Twitter Entity Sentiment (74k)",
      details: "Preserves hashtag content and idiomatic contractions."
    },
    {
      id: 3,
      title: "Senti-Pred Remake2 (Pipeline C)",
      category: "nlp-sentiment",
      categoryLabel: "NLP - Sentiment",
      status: "completed",
      description: "Voting Ensemble (LinearSVC+LR) with TF-IDF 100k and 4-grams. Record 97.80%.",
      techniques: ["TF-IDF 100k", "4-grams", "Voting Ensemble", "LinearSVC"],
      metric: {"label": "Accuracy", "value": "97.80%", "percent": 97.8},
      script: "experiments/nlp/twitter-entity-sentiment/senti-pred-variations/Senti-Pred-remake2/src/models/train.py",
      readme: "experiments/nlp/twitter-entity-sentiment/senti-pred-variations/README.md",
      models: ["Voting (LinearSVC+LogReg)"],
      dataset: "Twitter Sentiment (4 classes)",
      details: "Pipeline C: extreme vectorization with lemmatization and contraction expansion."
    },
    {
      id: 4,
      title: "Pipeline A vs B vs C (Head-to-Head + Ablations)",
      category: "nlp-sentiment",
      categoryLabel: "NLP - Sentiment",
      status: "completed",
      description: "Rigorous comparison of the 3 pipelines with ablation (n-grams, vocab, cleaning) and a McNemar test.",
      techniques: ["Ablation Study", "McNemar Test", "TF-IDF", "LinearSVC"],
      metric: {"label": "Best F1", "value": "0.9857", "percent": 98.57},
      script: "experiments/nlp/twitter-entity-sentiment/pipelines_abc_comparison/run_abc_comparison.ipynb",
      readme: "experiments/nlp/twitter-entity-sentiment/pipelines_abc_comparison/README.md",
      models: ["LinearSVC", "ExtraTrees", "LogReg"],
      dataset: "Twitter Sentiment",
      details: "Best F1 of the study: 0.9857 combining cleaning A + vectorizer C. Differences < 1pp are not significant."
    },
    {
      id: 5,
      title: "Ensemble Pyramid (6 Layers)",
      category: "nlp-sentiment",
      categoryLabel: "NLP - Sentiment",
      status: "completed",
      description: "Hierarchical pyramid with 6 layers of meta-ensembles (Bagging/Voting/Stacking).",
      techniques: ["Bagging", "Voting", "Stacking", "RL Meta-Learner", "TF-IDF 70k"],
      metric: {"label": "F1", "value": "~98%+", "percent": 98},
      script: "experiments/ensemble_pyramid.ipynb",
      readme: null,
      models: ["LR", "LinearSVC", "NB", "CNB", "Ridge", "RF", "ExtraTrees", "Meta-Stacking"],
      dataset: "Twitter Sentiment (4 classes)",
      details: "6 progressive layers. Lightweight PreFittedSoftVoting classes avoid retraining."
    },
    {
      id: 6,
      title: "Versatile Ensemble Pyramid (AutoML RL)",
      category: "nlp-sentiment",
      categoryLabel: "NLP - Sentiment",
      status: "completed",
      description: "AutoML engine with an RL Meta-Learner that decides the pyramid architecture dynamically.",
      techniques: ["RL Meta-Learner", "Thompson Sampling", "AutoML CLI", "TF-IDF"],
      metric: {"label": "AutoML", "value": "CLI", "percent": 95},
      script: "experiments/ensemble_pyramid.ipynb",
      readme: null,
      models: ["RL Agent", "Multi-Ensemble"],
      dataset: "Twitter Sentiment",
      details: "CLI with --layers, --strategy, --epsilon, --jitter. MLflow auto-tracking."
    },
    {
      id: 7,
      title: "Corrected Senti-Pred Pipeline",
      category: "nlp-sentiment",
      categoryLabel: "NLP - Sentiment",
      status: "completed",
      description: "Corrected pipeline with strict validation and removal of data leakage.",
      techniques: ["TF-IDF", "LinearSVC", "Leakage Prevention", "StratifiedKFold"],
      metric: {"label": "Accuracy", "value": "0.9810", "percent": 98.1},
      script: "experiments/nlp/twitter-entity-sentiment/senti_corrected.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["LinearSVC", "LogisticRegression"],
      dataset: "Twitter Entity Sentiment (74k)",
      details: "Stratified cross-validation and a decoupled pipeline for reliable production."
    },
    {
      id: 10,
      title: "Twitter Methods Comparison (5 Paradigms)",
      category: "nlp-class",
      categoryLabel: "NLP - Classification",
      status: "completed",
      description: "TF-IDF+LinearSVC vs DistilBERT vs TextCNN vs BiLSTM vs Sentence-BERT on the full dataset.",
      techniques: ["TF-IDF+LinearSVC", "DistilBERT", "TextCNN", "BiLSTM", "Sentence-BERT"],
      metric: {"label": "Best Acc", "value": "0.980", "percent": 98},
      script: "experiments/nlp/twitter-entity-sentiment/NLP-twitter-methods-comparasion.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["LinearSVC", "DistilBERT", "TextCNN", "BiLSTM", "Sentence-BERT"],
      dataset: "Twitter (74k)",
      details: "TF-IDF+LinearSVC 0.98 in 4.35s. DistilBERT 0.971 in 40min. TextCNN has the best neural cost-benefit ratio."
    },
    {
      id: 11,
      title: "AG News Classification (Low-Data)",
      category: "nlp-class",
      categoryLabel: "NLP - Classification",
      status: "completed",
      description: "DistilBERT fine-tune vs TF-IDF on 1k samples. The Transformer wins in low-data.",
      techniques: ["DistilBERT", "TF-IDF", "Grid Search", "Fine-tuning"],
      metric: {"label": "Accuracy", "value": "0.835", "percent": 83.5},
      script: "experiments/nlp/ag-news-classification.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["DistilBERT", "LinearSVC", "ExtraTrees"],
      dataset: "AG News (4 classes, 1k train)",
      details: "DistilBERT 0.835 vs TF-IDF+LinearSVC 0.765. Grid search is optimal at 3-4k features."
    },
    {
      id: 12,
      title: "Multi-Task Learning (MMoE)",
      category: "nlp-class",
      categoryLabel: "NLP - Classification",
      status: "completed",
      description: "MMoE with go_emotions. TF-IDF 15k + Focal Loss + ExtraTrees beats neural networks.",
      techniques: ["MMoE", "Focal Loss", "ExtraTrees", "TF-IDF 15k", "Deep Learning"],
      metric: {"label": "F1-weighted", "value": "0.9643", "percent": 96.4},
      script: "experiments/nlp/nlp-multi-task-classification.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["MMoE Neural", "ExtraTrees", "LinearSVC", "LightGBM"],
      dataset: "go_emotions (43k)",
      details: "ExtraTrees 0.9643 beats all the networks. MMoE+Focal Loss 0.9566 with sparse features."
    },
    {
      id: 13,
      title: "Logistic Regression Multiclass Strategies",
      category: "nlp-class",
      categoryLabel: "NLP - Classification",
      status: "completed",
      description: "Multinomial vs OvR vs OvO with several solvers and values of C.",
      techniques: ["Logistic Regression", "Multinomial", "OvR", "OvO", "TF-IDF"],
      metric: {"label": "Best Acc", "value": "0.982", "percent": 98.2},
      script: "experiments/nlp/twitter-entity-sentiment/logistic-regression-multiclass.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["Multinomial(lbfgs)", "OvR(saga)", "OvO(liblinear)"],
      dataset: "Twitter Sentiment",
      details: "Max diff between strategies: 0.4pp. Multinomial lbfgs C=10 wins."
    },
    {
      id: 14,
      title: "Feature Engineering NLP",
      category: "nlp-class",
      categoryLabel: "NLP - Classification",
      status: "completed",
      description: "Hashing trick beats TF-IDF (0.986 vs 0.977). Word+char n-grams add +0.5pp.",
      techniques: ["Hashing Trick", "TF-IDF", "Word+Char N-grams", "Domain Features"],
      metric: {"label": "Best Acc", "value": "0.986", "percent": 98.6},
      script: "experiments/nlp/twitter-entity-sentiment/feature-engineering-nlp.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["LinearSVC", "ExtraTrees"],
      dataset: "Twitter Sentiment",
      details: "Hashing trick: 262k features with no IDF cost. Trees only gain from domain knowledge."
    },
    {
      id: 16,
      title: "NLP Regression — Wine Scores",
      category: "nlp-class",
      categoryLabel: "NLP - Classification",
      status: "completed",
      description: "Ridge vs LightGBM predicting wine scores from text. The linear model wins on sparse data.",
      techniques: ["TF-IDF 15k", "Ridge Regression", "LightGBM", "MLflow"],
      metric: {"label": "MAE", "value": "1.13", "percent": 88},
      script: "experiments/nlp-regression-wine/nlp_regression_wine.ipynb",
      readme: "experiments/nlp-regression-wine/README.md",
      models: ["Ridge Regressor", "LightGBM Regressor"],
      dataset: "Wine Reviews (Kaggle)",
      details: "Ridge MAE 1.13 R2 0.69 vs LightGBM MAE 1.47 R2 0.63. Linear thrives in sparse space."
    },
    {
      id: 40,
      title: "Hierarchical Classification (20 Newsgroups)",
      category: "hierarchical",
      categoryLabel: "Hierarchical",
      status: "completed",
      description: "Flat vs hierarchical local-per-node with TF-IDF word+char and LinearSVC.",
      techniques: ["TF-IDF word+char", "LinearSVC", "Hierarchical Classifier", "20 Newsgroups"],
      metric: {"label": "Flat Acc", "value": "0.7188", "percent": 71.9},
      script: "experiments/hierarchical/hierarchical_classification.ipynb",
      readme: "experiments/hierarchical/README.md",
      models: ["LinearSVC(C=0.15)", "LogisticRegression"],
      dataset: "20 Newsgroups (18k docs, 20 classes)",
      details: "Flat 0.7188 vs Hierarchical 0.6953. HF: 0.7668 vs 0.7516. The parent node is the bottleneck."
    },
    {
      id: 41,
      title: "Clustering: Flat vs Hierarchical",
      category: "hierarchical",
      categoryLabel: "Hierarchical",
      status: "completed",
      description: "KMeans vs Agglomerative vs 2-level top-down. Top-down wins at leaf level.",
      techniques: ["KMeans", "Agglomerative(Ward)", "Top-down Clustering", "TF-IDF+SVD"],
      metric: {"label": "Top-down NMI", "value": "0.360", "percent": 72},
      script: "experiments/hierarchical/clustering_flat_vs_hierarchical.ipynb",
      readme: "experiments/hierarchical/README.md",
      models: ["KMeans", "AgglomerativeClustering"],
      dataset: "20 Newsgroups (3k sample)",
      details: "Top-down Purity 0.398, NMI 0.360 — above the literature (0.25-0.45)."
    },
    {
      id: 45,
      title: "Evolutionary Feature Selection (GAAP/MO-DE)",
      category: "feature-selection",
      categoryLabel: "Feature Selection",
      status: "completed",
      description: "Multi-objective NSGA-II and Differential Evolution vs SelectKBest/Boruta/RF importance.",
      techniques: ["NSGA-II", "MO-DE", "DEAP", "SelectKBest", "Boruta", "Pareto Front"],
      metric: {"label": "R2 (23 feats)", "value": "0.694", "percent": 82},
      script: "experiments/feature_selection_ea/feature_selection_ea.ipynb",
      readme: "experiments/feature_selection_ea/README.md",
      models: ["Ridge", "LogisticRegression", "GAAP", "MO-DE"],
      dataset: "California Housing + Twitter",
      details: "EA wins on interactive features (California). Classical methods suffice for bag-of-words."
    },
    {
      id: 50,
      title: "CV Methods Comparison (CIFAR-10)",
      category: "cv",
      categoryLabel: "Computer Vision",
      status: "completed",
      description: "HOG+SVM vs ResNet18 vs ViT on CIFAR-10. ViT reaches 0.9805.",
      techniques: ["HOG+SVM", "ResNet18", "ViT", "Fine-tuning", "ImageNet-21k"],
      metric: {"label": "ViT Acc", "value": "0.9805", "percent": 98.1},
      script: "experiments/computer_vision/cv-methods-comparison.ipynb",
      readme: "experiments/computer_vision/README.md",
      models: ["HOG+SVM", "ResNet18", "ViT-base-patch16-224"],
      dataset: "CIFAR-10 (50k/10k)",
      details: "ViT 0.9805 > ResNet18 0.9362 > HOG 0.3970. Manual features fail at low resolution."
    },
    {
      id: 51,
      title: "Animal Multi-Label (4 Approaches)",
      category: "cv",
      categoryLabel: "Computer Vision",
      status: "completed",
      description: "ResNet18+Aug, VGG16, CLIP zero-shot, EfficientNet-B0 for multi-label pet classification.",
      techniques: ["ResNet18", "VGG16", "CLIP Zero-shot", "EfficientNet", "BCEWithLogitsLoss"],
      metric: {"label": "ResNet F1", "value": "1.000", "percent": 100},
      script: "experiments/computer_vision/animal-classifier.ipynb",
      readme: "experiments/computer_vision/README.md",
      models: ["ResNet18+Aug", "VGG16", "CLIP", "EfficientNet-B0"],
      dataset: "Pet Images (44, 2 classes)",
      details: "ResNet18+Aug is perfect (F1 1.000). CLIP requires careful threshold calibration."
    },
    {
      id: 52,
      title: "Face Recognition App",
      category: "cv",
      categoryLabel: "Computer Vision",
      status: "completed",
      description: "Face recognition app with 3 modes: LBPH, CNN and Transfer Learning (YuNet+MobileNetV2).",
      techniques: ["LBPH", "CNN", "YuNet", "MobileNetV2", "Face Detection"],
      metric: null,
      script: "experiments/computer_vision/face_recognition_app.ipynb",
      readme: "experiments/computer_vision/README.md",
      models: ["LBPH", "CNN", "MobileNetV2+YuNet"],
      dataset: "Local Face Dataset",
      details: "Collection via upload, training and prediction. LBPH runs on CPU."
    },
    {
      id: 53,
      title: "YOLO Object Detection",
      category: "cv",
      categoryLabel: "Computer Vision",
      status: "completed",
      description: "Object detection via YOLOv3-tiny COCO using OpenCV DNN.",
      techniques: ["YOLOv3-tiny", "OpenCV DNN", "Object Detection"],
      metric: null,
      script: "experiments/computer_vision/yolo_notebook.ipynb",
      readme: "experiments/computer_vision/README.md",
      models: ["YOLOv3-tiny"],
      dataset: "COCO / Custom",
      details: "Image upload and one-stage detection. Supports custom models."
    },
    {
      id: 54,
      title: "Knowledge Distillation — CIFAR-10 (3 Families)",
      category: "cv",
      categoryLabel: "Computer Vision",
      status: "completed",
      description: "Response-based (Logit), Feature-based (FitNets), Relation-based (RKD) and Hybrid KD.",
      techniques: ["Logit KD", "FitNets", "RKD", "ResNet18 Teacher", "CNN Student"],
      metric: null,
      script: "experiments/computer_vision/kd-cifar10-comparison.ipynb",
      readme: "experiments/computer_vision/README.md",
      models: ["ResNet18 (Teacher)", "CNN 1.1M (Student)"],
      dataset: "CIFAR-10 (50k/10k)",
      details: "Teacher ResNet18 fine-tuned 224x224 (~0.91). Student CNN ~10x smaller. 5 scenarios compared."
    },
    {
      id: 55,
      title: "MovieLens RecSys (8 Paradigms)",
      category: "recsys",
      categoryLabel: "Recommender Systems",
      status: "completed",
      description: "Popularity, KNN User/Item, SVD, NCF, Two-Tower, LightGBM+FE, BPR on MovieLens 100k.",
      techniques: ["SVD", "KNN", "NCF", "Two-Tower", "LightGBM", "BPR", "Matrix Factorization"],
      metric: {"label": "Two-Tower RMSE", "value": "0.9297", "percent": 93},
      script: "experiments/recommender_systems/movielens-recsys.ipynb",
      readme: "experiments/recommender_systems/README.md",
      models: ["SVD", "NCF", "Two-Tower", "LightGBM", "BPR", "KNN"],
      dataset: "MovieLens 100k (93.7% sparsity)",
      details: "Two-Tower 0.9297. Cold-start demo: SVD recommends Empire Strikes Back for a Star Wars profile."
    },
    {
      id: 56,
      title: "MovieLens AutoRec (10 Models)",
      category: "recsys",
      categoryLabel: "Recommender Systems",
      status: "completed",
      description: "Item-AutoRec beats all 10 models with RMSE 0.9054. Autoencoder with masked MSE.",
      techniques: ["AutoRec", "Autoencoder", "Masked MSE", "Collaborative Filtering"],
      metric: {"label": "Item-AutoRec RMSE", "value": "0.9054", "percent": 91},
      script: "experiments/recommender_systems/movielens-autorec.ipynb",
      readme: "experiments/recommender_systems/README.md",
      models: ["Item-AutoRec", "User-AutoRec", "SVD", "Two-Tower", "NCF"],
      dataset: "MovieLens 100k",
      details: "MLflow tracked. Item-based >> User-based at high sparsity."
    },
    {
      id: 57,
      title: "Image Recommender (Visual Similarity)",
      category: "recsys",
      categoryLabel: "Recommender Systems",
      status: "completed",
      description: "Recommendation by visual similarity: ResNet embeddings + cosine similarity.",
      techniques: ["ResNet Embeddings", "Cosine Similarity", "L2 Normalization"],
      metric: null,
      script: "experiments/recommender_systems/image_recommender.ipynb",
      readme: "experiments/recommender_systems/README.md",
      models: ["ResNet Feature Extractor"],
      dataset: "Local Images (~30)",
      details: "Pipeline: collect -> embed -> L2 normalize -> cosine top-K. 30 images indexed in 4.2s."
    },
    {
      id: 60,
      title: "Prophet + Optuna (Temperature)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Bayesian tuning of Prophet with Optuna minimizing MAE in Time Series CV.",
      techniques: ["Prophet", "Optuna", "Bayesian Optimization", "Cross Validation"],
      metric: {"label": "MAE", "value": "1.96", "percent": 85},
      script: "experiments/time_series/temperature_forecasting_prophet.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["Prophet (Optuna)"],
      dataset: "Daily Min Temperatures",
      details: "Multiplicative seasonality. changepoint_prior_scale and seasonality_prior_scale tuned."
    },
    {
      id: 61,
      title: "Prophet vs LightGBM (Temperature)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "LightGBM with lags+rolling beats Prophet on abrupt daily noise.",
      techniques: ["LightGBM", "Prophet", "Lag Features", "Rolling Windows"],
      metric: {"label": "LGBM MAE", "value": "1.7344", "percent": 90},
      script: "experiments/time_series/temperature_forecasting_prophet.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["LightGBM", "Prophet"],
      dataset: "Daily Temperatures",
      details: "LightGBM 1.7344 vs Prophet 1.96. Trees react better to abrupt noise."
    },
    {
      id: 62,
      title: "Sales Forecast V2.2 (Hackathon)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "LightGBM with 32 features + Optuna pruning + MLflow + Docker + 10 Pytest tests.",
      techniques: ["LightGBM", "Optuna", "MLflow", "Docker", "Pytest", "32 Features"],
      metric: {"label": "MAE", "value": "1.4218", "percent": 95},
      script: "experiments/sales-forecast/Predictive_Sales_Pipeline.ipynb",
      readme: "experiments/sales-forecast/README.md",
      models: ["LightGBM Regressor"],
      dataset: "Sales 2022 (5.6M rows)",
      details: "V2->V2.2: MAE 2.5769->1.4218 (-44.8%). 10 high-cardinality categorical features."
    },
    {
      id: 63,
      title: "AE Embedding Experiments (Sales)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Autoencoder embeddings as features, plus clustering for sales forecasting. Everything gets worse.",
      techniques: ["Autoencoder", "MLP", "Causal Mask", "K-means Clustering"],
      metric: {"label": "Causal AE", "value": "-0.14%", "percent": 50},
      script: "experiments/sales-forecast/ae_embedding_experiments.ipynb",
      readme: "experiments/sales-forecast/README.md",
      models: ["MLP Autoencoder (47->8)"],
      dataset: "Sales 709k series",
      details: "Causal AE is neutral (-0.14%). Naive has leakage (+20%). Clustering makes everything worse."
    },
    {
      id: 64,
      title: "Decomposition vs Regression (Sales)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Tests decomposition (level+trend+seasonality) vs LightGBM regression.",
      techniques: ["Decomposition", "LightGBM", "Time Series"],
      metric: null,
      script: "experiments/sales-forecast/decomposition_vs_regression.ipynb",
      readme: "experiments/sales-forecast/README.md",
      models: ["LightGBM", "Decomposition"],
      dataset: "Sales 5.6M",
      details: "Supervised regression beats classical decomposition."
    },
    {
      id: 65,
      title: "Knowledge Distillation (Time Series)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "LSTM+Attention->TCN: the Student retains 103.9% of the Teacher. LGBM->LGBM fails.",
      techniques: ["LSTM", "TCN", "Attention", "Knowledge Distillation"],
      metric: {"label": "Student-KD", "value": "103.9%", "percent": 99},
      script: "experiments/time_series/knowledge_distillation-time_series.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["LSTM+Attn (1.44M)", "TCN (228k)"],
      dataset: "Hourly Electricity",
      details: "KD works for dense networks. It fails for trees (the teacher overfits)."
    },
    {
      id: 66,
      title: "Anomaly Detection (5 Techniques)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Z-Score wins (F1 0.9954, 0 false alarms) on Melbourne temperature.",
      techniques: ["Z-Score", "Prophet Intervals", "Isolation Forest", "Elliptic Envelope", "LOF"],
      metric: {"label": "Z-Score F1", "value": "0.9954", "percent": 99.5},
      script: "experiments/time_series/exp4_anomaly_detection.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["Z-Score", "IsolationForest", "Prophet", "EE", "LOF"],
      dataset: "Melbourne Temp (3650 days)",
      details: "Z-Score: 108/109 anomalies, 0 false positives. The Prophet 99.9% interval is also excellent."
    },
    {
      id: 67,
      title: "Benchmark 4x4 (Paradigms x Scenarios)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "SARIMA, Prophet, TCN, LightGBM on 4 datasets. SARIMA wins 2/4.",
      techniques: ["SARIMA", "Prophet", "TCN", "LightGBM", "Diebold-Mariano"],
      metric: null,
      script: "experiments/time_series/benchmark-ts-paradigms.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["SARIMA", "Prophet", "TCN", "LightGBM"],
      dataset: "CO2, Nile, Sunspots, Synthetic",
      details: "SARIMA 1st on CO2/Nile. TCN 1st on Sunspots. Prophet 1st on Synthetic. DM test p<0.05 in 3/4."
    },
    {
      id: 68,
      title: "TS Classification (6 Paradigms)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "ROCKET dominates 3/3 UEA datasets. The DTW baseline is robust but slow.",
      techniques: ["ROCKET", "1-NN+DTW", "InceptionTime", "TSFresh+RF", "Transformer", "LightGBM+FE"],
      metric: {"label": "ROCKET GunPoint", "value": "1.000", "percent": 100},
      script: "experiments/time_series/time-series-classification.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["ROCKET", "DTW", "InceptionTime", "Transformer"],
      dataset: "GunPoint/ArrowHead/ECG5000",
      details: "ROCKET: 1.000/0.953/0.889. DTW: 36min on ECG5000. The Transformer collapses on ArrowHead."
    },
    {
      id: 69,
      title: "TS + NLP (Stock Sentiment Fusion)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Fuses time series with NLP for synthetic market direction prediction.",
      techniques: ["LightGBM", "TS+NLP Fusion", "Sentiment Features"],
      metric: {"label": "NLP-only Acc", "value": "0.730", "percent": 73},
      script: "experiments/time_series/stock-sentiment-ts-nlp.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["LightGBM (TS/NLP/TS+NLP)"],
      dataset: "Synthetic GBM + Headlines",
      details: "NLP-only 0.730, TS+NLP F1 0.720, TS-only 0.492. The lagged news item dominates."
    },
    {
      id: 70,
      title: "Forecast -> Direction Classification",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Converts the forecast into direction classification (up/down). Logistic wins.",
      techniques: ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
      metric: {"label": "Logistic Acc", "value": "0.958", "percent": 95.8},
      script: "experiments/time_series/forecast-classification.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["Logistic", "RF", "XGBoost", "LightGBM"],
      dataset: "fato_vendas (daily agg)",
      details: "Logistic Acc 0.958, AUC 0.967. is_weekend, dow, lag_7 = 59% importance."
    },
    {
      id: 71,
      title: "TS Feature Engineering Phase 1 — Manual vs tsfresh",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Manual feature engineering (lags, rolling) vs automatic (tsfresh) on daily temperature.",
      techniques: ["tsfresh", "Manual FE", "Rolling Windows", "Lags", "Random Forest"],
      metric: {"label": "Manual MAE", "value": "1.74", "percent": 90},
      script: "experiments/ts_fe/automated_vs_manual_fe_ts.ipynb",
      readme: "experiments/ts_fe/README.md",
      models: ["Random Forest", "LightGBM"],
      dataset: "Daily Min Temperatures",
      details: "Manual features reduce dimensionality and beat tsfresh in speed and stability."
    },
    {
      id: 72,
      title: "TS Feature Engineering Phase 2 — Multivariate (Beijing PM2.5)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Multivariate feature engineering on 5 simultaneous weather variables (Temperature, Wind, Pressure).",
      techniques: ["Multivariate FE", "tsfresh", "Cross-Variable Lags", "Random Forest"],
      metric: {"label": "Manual MAE", "value": "56.80", "percent": 88},
      script: "experiments/ts_fe/multivariate_auto_vs_manual_fe.ipynb",
      readme: "experiments/ts_fe/README.md",
      models: ["Random Forest", "LightGBM"],
      dataset: "Beijing PM2.5 Air Quality",
      details: "Crossed lags capture the weather dynamics before pollution peaks."
    },
    {
      id: 73,
      title: "TS Feature Engineering Phase 3 — Deep Learning Embeddings",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "LSTM Autoencoder in PyTorch to extract latent representations and compress time windows.",
      techniques: ["LSTM Autoencoder", "PyTorch", "Latent Embeddings", "Representation Learning"],
      metric: {"label": "Bottleneck", "value": "16 dims", "percent": 85},
      script: "experiments/ts_fe/dl_embeddings_fe_ts.ipynb",
      readme: "experiments/ts_fe/README.md",
      models: ["LSTM Autoencoder", "Random Forest"],
      dataset: "Beijing PM2.5",
      details: "Compression of 35 features into 16 latent dimensions to feed the decision trees."
    },
    {
      id: 74,
      title: "TS Feature Engineering Phase 4 — Signals & Wavelets (DWT)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Discrete Wavelet Transform (DWT) and Seasonal Decomposition without temporal leakage.",
      techniques: ["Discrete Wavelet (DWT)", "Seasonal Decompose", "Shifted Windows", "Signal Processing"],
      metric: {"label": "DWT MAE", "value": "54.19", "percent": 95},
      script: "experiments/ts_fe/advanced_signal_fe_ts.ipynb",
      readme: "experiments/ts_fe/README.md",
      models: ["Random Forest", "Wavelet Filters"],
      dataset: "Beijing PM2.5",
      details: "DWT on sliding windows breaks the previous record (MAE 54.19 vs 56.80)."
    },
    {
      id: 75,
      title: "TS Feature Engineering Phase 5 — Time Embeddings & HPO",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Trigonometric time embeddings (sine/cosine) and Bayesian optimization via Optuna.",
      techniques: ["Time Embeddings", "Optuna", "Bayesian Optimization", "Cyclical Features"],
      metric: {"label": "Optuna MAE", "value": "53.80", "percent": 96},
      script: "experiments/ts_fe/hpo_time_embeddings_ts.ipynb",
      readme: "experiments/ts_fe/README.md",
      models: ["Random Forest Tuned", "Optuna TPE"],
      dataset: "Beijing PM2.5",
      details: "Temporal harmonics remove the discontinuity between the end and the start of annual cycles."
    },
    {
      id: 76,
      title: "Sktime vs Hybrid (Phase 6)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "sktime WindowSummarizer vs a hybrid model with Wavelets on Beijing PM2.5.",
      techniques: ["sktime", "WindowSummarizer", "Wavelets", "Random Forest"],
      metric: {"label": "MAE", "value": "52.79", "percent": 99},
      script: "experiments/time_series/sktime_vs_hybrid_ts.ipynb",
      readme: "experiments/ts_fe/README.md",
      models: ["Random Forest", "sktime Summarizer"],
      dataset: "Beijing PM2.5",
      details: "sktime WindowSummarizer breaks the record with MAE 52.79 in under 1s of runtime."
    },
    {
      id: 77,
      title: "Multivariate Time Series — VAR (Vector Autoregression)",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Simultaneous modeling of GDP, Consumption, Investment and Inflation via VAR with AIC/BIC selection.",
      techniques: ["VAR", "Granger Causality", "Impulse Response (IRF)", "Cointegration", "Stationarity"],
      metric: {"label": "Best VAR Order", "value": "p=2", "percent": 90},
      script: "experiments/time_series/multivariate-time-series-var.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["Vector Autoregression (VAR)", "ARIMA"],
      dataset: "US Macrodata (1959-2009)",
      details: "Captures interconnected shocks (IRF). Multivariate modeling beats a standalone univariate ARIMA."
    },
    {
      id: 78,
      title: "Property Sales — SARIMAX + Box-Cox + Prophet",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Real estate sales forecasting with Box-Cox transformation, exogenous variables and Prophet.",
      techniques: ["SARIMAX", "Box-Cox", "Exogenous Variables", "Prophet", "Expanding Window CV"],
      metric: {"label": "MAPE", "value": "4.21%", "percent": 94},
      script: "experiments/time_series/property-sales-time-series.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["SARIMAX Exog", "Prophet", "Auto-ARIMA"],
      dataset: "Property Sales Time Series",
      details: "Box-Cox stabilizes severe heteroscedasticity. Multivariate SARIMAX reduces error by 38%."
    },
    {
      id: 79,
      title: "Sales Forecast — Integrated sktime Forecaster",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Training and inference module for Sales Forecast using sktime pipelines.",
      techniques: ["sktime", "DirectTabularRegressionForecaster", "LightGBM", "Recursive Forecast"],
      metric: {"label": "Speed", "value": "Instant", "percent": 95},
      script: "experiments/sales-forecast/scripts/train_sktime.py",
      readme: "experiments/sales-forecast/README.md",
      models: ["LightGBM + sktime Forecaster"],
      dataset: "fato_vendas (Aggregated)",
      details: "Object-oriented implementation (OOP) ready for microservice integration."
    },
    {
      id: 601,
      title: "Hierarchical Forecasting — Bottom-Up Reconciliation",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "Hierarchical time-series forecasting with bottom-up reconciliation vs a direct forecast of the Total (Total -> Regions -> Leaves).",
      techniques: ["Hierarchical Forecasting", "Bottom-Up Reconciliation", "Seasonal Naive", "Aggregation Matrix", "MAPE"],
      metric: {"label": "Coherence", "value": "100%", "percent": 99},
      script: "experiments/time_series/hierarchical_forecast.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["Seasonal Naive", "Bottom-Up Reconciler", "Direct Total Forecast"],
      dataset: "Hierarchical Synthetic Demand",
      details: "Hierarchical structure in 3 levels (Total -> Regions R1/R2 -> Leaves A1/A2/B1/B2). Bottom-up reconciliation guarantees additive coherence across the tree."
    },
    {
      id: 80,
      title: "Anomaly: Supervised vs Unsupervised",
      category: "anomaly",
      categoryLabel: "Anomaly Detection",
      status: "completed",
      description: "Random Forest vs Isolation Forest on NAB machine temperature.",
      techniques: ["Random Forest", "Isolation Forest", "Feature Engineering"],
      metric: {"label": "RF F1", "value": "0.991", "percent": 99.1},
      script: "experiments/anomaly_detection_comparison.ipynb",
      readme: null,
      models: ["RandomForestClassifier", "IsolationForest"],
      dataset: "NAB Machine Temp",
      details: "Supervised vs unsupervised comparison for anomaly detection."
    },
    {
      id: 81,
      title: "Anomaly: 4 Paradigms Enhanced",
      category: "anomaly",
      categoryLabel: "Anomaly Detection",
      status: "completed",
      description: "Density, Clustering, Representation Learning (Autoencoder), Binary Classification.",
      techniques: ["Isolation Forest", "LOF", "OCSVM", "GMM", "KMeans", "DBSCAN", "Autoencoder", "XGBoost", "SMOTE"],
      metric: {"label": "Best F1", "value": "0.9954", "percent": 99.5},
      script: "experiments/anomaly_detection_enhanced.ipynb",
      readme: null,
      models: ["IF", "LOF", "OCSVM", "EE", "GMM", "KMeans", "DBSCAN", "AE", "RF+SMOTE", "XGBoost"],
      dataset: "NAB Machine Temp (22.7k)",
      details: "4 families: density, clustering, representation learning, binary classification."
    },
    {
      id: 82,
      title: "Optimized Anomaly Detection Time Series",
      category: "anomaly",
      categoryLabel: "Anomaly Detection",
      status: "completed",
      description: "Optimized pipeline for anomaly detection in time series with threshold tuning.",
      techniques: ["Isolation Forest", "Z-Score", "Threshold Tuning", "Rolling Statistics"],
      metric: {"label": "Precision", "value": "99.2%", "percent": 99.2},
      script: "experiments/time_series/anomaly_detection_optimized.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["Random Forest", "Isolation Forest"],
      dataset: "NAB Machine Temperature",
      details: "Calibration of statistical thresholds to minimize false positives in monitoring."
    },
    {
      id: 85,
      title: "Clustering: Unsupervised vs Semi vs Supervised",
      category: "clustering",
      categoryLabel: "Clustering",
      status: "completed",
      description: "KMeans, DBSCAN, Agglomerative, GMM vs RF/KNN on Iris, Blobs, Moons, Circles.",
      techniques: ["KMeans", "DBSCAN", "Agglomerative", "GMM", "RF", "KNN"],
      metric: {"label": "ARI", "value": "0.92", "percent": 92},
      script: "experiments/hierarchical/clustering_comparison.ipynb",
      readme: "experiments/hierarchical/README.md",
      models: ["KMeans", "DBSCAN", "AgglomerativeClustering", "GMM"],
      dataset: "Iris/Blobs/Moons/Circles",
      details: "Compares ARI/NMI between unsupervised and supervised approaches."
    },
    {
      id: 86,
      title: "Supervised Clustering (Concept)",
      category: "clustering",
      categoryLabel: "Clustering",
      status: "completed",
      description: "The supervised clustering concept: clustering guided by labels, evaluated by ARI/NMI.",
      techniques: ["KMeans+LDA", "GMM+Labels", "PCA", "RF-guided"],
      metric: {"label": "NMI", "value": "0.89", "percent": 89},
      script: "experiments/hierarchical/supervised_clustering.ipynb",
      readme: "experiments/hierarchical/README.md",
      models: ["KMeans", "LDA", "GMM"],
      dataset: "Iris",
      details: "Cluster_id has no meaning; what matters is the grouping."
    },
    {
      id: 87,
      title: "Senti-Pred as Supervised Clustering",
      category: "clustering",
      categoryLabel: "Clustering",
      status: "completed",
      description: "Applies supervised clustering to the Twitter Sentiment dataset with TF-IDF.",
      techniques: ["TF-IDF", "PCA", "TruncatedSVD", "LDA", "KMeans", "DBSCAN"],
      metric: {"label": "SVD+LDA ARI", "value": "0.78", "percent": 78},
      script: "experiments/nlp/twitter-entity-sentiment/senti_supervised_clustering.ipynb",
      readme: "experiments/nlp/README.md",
      models: ["KMeans", "LDA", "DBSCAN", "RF"],
      dataset: "Twitter Sentiment",
      details: "Dimensionality reduction (PCA/SVD/LDA) + clustering on textual data."
    },
    {
      id: 88,
      title: "Forecast: Supervised vs Statistical",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "RF, SARIMAX, Prophet, ExpSmoothing, XGBoost on Air Passengers and Sunspots.",
      techniques: ["Random Forest", "SARIMAX", "Prophet", "ExponentialSmoothing", "XGBoost"],
      metric: {"label": "Prophet MAPE", "value": "3.8%", "percent": 96},
      script: "experiments/time_series/forecast_comparison.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["RF", "SARIMAX", "Prophet", "ExpSmoothing", "XGBoost"],
      dataset: "Air Passengers + Sunspots",
      details: "Compares supervised (lag features) vs statistical approaches."
    },
    {
      id: 90,
      title: "Feature Engineering Tabular (10 Techniques)",
      category: "regression",
      categoryLabel: "Tabular Regression",
      status: "completed",
      description: "10 FE techniques on California Housing with LR, LightGBM, RF. Asymmetric FE per model.",
      techniques: ["Polynomial", "PCA", "Geo Features", "Log Transform", "Binning", "Standardization"],
      metric: {"label": "Best R2", "value": "0.8418", "percent": 84.2},
      script: "experiments/tabular_regression/feature-engineering-tabular.ipynb",
      readme: "experiments/tabular_regression/README.md",
      models: ["LinearRegression", "LightGBM", "RandomForest"],
      dataset: "California Housing (20.6k)",
      details: "Combined +13.5pp for LR. Geo +0.6pp for LGBM. PCA -9 to -18pp for all."
    },
    {
      id: 91,
      title: "Price Prediction v1->v3",
      category: "regression",
      categoryLabel: "Tabular Regression",
      status: "completed",
      description: "Evolution of the car price prediction pipeline up to R2 0.9489 (Random Forest).",
      techniques: ["Random Forest", "GridSearchCV", "log1p Target", "One-Hot Encoding"],
      metric: {"label": "R2", "value": "0.9489", "percent": 94.9},
      script: "experiments/tabular_regression/price-prediction-multiple-linear-regression.ipynb",
      readme: "experiments/tabular_regression/README.md",
      models: ["RandomForest", "XGBoost", "ElasticNet", "Ridge"],
      dataset: "Car Prices (205 samples)",
      details: "v2: R2 0.8517->0.9489. v3 plateau (limiting dataset). Residuals are normal."
    },
    {
      id: 92,
      title: "Watsonx Local AutoML Equivalent",
      category: "regression",
      categoryLabel: "Tabular Regression",
      status: "completed",
      description: "FLAML, TPOT, 9 baselines on California Housing. XGBoost beats AutoML.",
      techniques: ["FLAML", "TPOT", "XGBoost", "AutoML", "Ridge", "Lasso", "SVR"],
      metric: {"label": "XGB R2", "value": "0.8401", "percent": 84},
      script: "experiments/tabular_regression/ibm-watsonx-local-automl.ipynb",
      readme: "experiments/ibm-experiments/README.md",
      models: ["XGBoost", "FLAML(CatBoost)", "TPOT", "ExtraTrees"],
      dataset: "California Housing",
      details: "Manual XGBoost beats FLAML/TPOT by a small margin. AutoML = good baseline."
    },
    {
      id: 93,
      title: "California Housing — Advanced Regression Strategies",
      category: "regression",
      categoryLabel: "Tabular Regression",
      status: "completed",
      description: "Comparative study: Univariate, Multiple, Ridge+RFE, Random Forest and Gradient Boosting.",
      techniques: ["Haversine Geo FE", "RFE Selection", "Gradient Boosting", "Random Forest", "VIF Analysis"],
      metric: {"label": "GB R2", "value": "0.8034", "percent": 80.3},
      script: "experiments/tabular_regression/california-house-regression.ipynb",
      readme: "experiments/tabular_regression/README.md",
      models: ["GradientBoostingRegressor", "RandomForestRegressor", "Ridge+RFE", "LinearRegression"],
      dataset: "California Housing (20.6k)",
      details: "Geographic feature engineering (distance to economic hubs) improves R2 from 0.60 to 0.8034."
    },
    {
      id: 95,
      title: "IBM Watsonx — Boston Housing (Cloud)",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "external",
      description: "Original Watsonx AutoAI notebook for regression. Requires cloud credentials.",
      techniques: ["Snap ML", "AutoAI", "IBM Watson"],
      metric: null,
      script: "experiments/ibm-experiments/Boston Housing Price Prediction.ipynb",
      readme: "experiments/ibm-experiments/README.md",
      models: ["Snap ML Regressor"],
      dataset: "Boston Housing",
      details: "Original cloud version. Local equivalent: tabular_regression/ibm-watsonx-local-automl.ipynb."
    },
    {
      id: 96,
      title: "IBM Watsonx — Electric Production (Cloud)",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "external",
      description: "Original autoai-ts-libs notebook for forecasting. Requires cloud credentials.",
      techniques: ["autoai-ts-libs", "IBM Watson", "Time Series"],
      metric: null,
      script: "experiments/ibm-experiments/Electric_Production.ipynb",
      readme: "experiments/ibm-experiments/README.md",
      models: ["Snap ML Forecaster"],
      dataset: "Electric Production",
      details: "Original cloud version. Local equivalent: time_series/ibm-watsonx-local-timeseries.ipynb."
    },
    {
      id: 97,
      title: "IBM Watsonx — Sentiment (Cloud)",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "external",
      description: "Original Watsonx sentiment analysis. Metrics TBD.",
      techniques: ["Snap ML", "Sentiment Analysis", "NLP"],
      metric: null,
      script: "experiments/ibm-experiments/sentiment_analysis.ipynb",
      readme: "experiments/ibm-experiments/README.md",
      models: ["Snap ML Classifier"],
      dataset: "Sentiment Dataset",
      details: "Original cloud version. Metrics depend on execution in Watsonx."
    },
    {
      id: 98,
      title: "IBM AutoAI — Electric Production Pipeline Assembler",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "external",
      description: "Scikit-learn representation of the pipeline exported by Watsonx AutoAI for time series.",
      techniques: ["autoai-ts-libs", "Scikit-Learn Pipeline", "Holdout Evaluation", "IBM Cloud Deployment"],
      metric: null,
      script: "experiments/ibm-experiments/_P4 - Assembler_ Electric_Production.ipynb",
      readme: "experiments/ibm-experiments/README.md",
      models: ["AutoAI Exported Pipeline"],
      dataset: "Electric Production",
      details: "Scoring and forecasting pipeline generated with the official Scikit-Learn operations definition."
    },
    {
      id: 99,
      title: "IBM AutoAI — Snap Random Forest Classifier",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "external",
      description: "Classification pipeline with Snap ML acceleration exported from IBM Watsonx.",
      techniques: ["Snap ML", "SnapRandomForestClassifier", "AutoAI Pipeline"],
      metric: null,
      script: "experiments/ibm-experiments/_P5 - Snap random forest classifier....ipynb",
      readme: "experiments/ibm-experiments/README.md",
      models: ["SnapRandomForestClassifier"],
      dataset: "Classification Benchmark",
      details: "Implementation with native Snap ML acceleration for ultra-fast inference."
    },
    {
      id: 101,
      title: "IBM AutoAI — Snap Random Forest Regressor",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "external",
      description: "Regression pipeline with the Snap ML Random Forest Regressor exported from IBM Watsonx.",
      techniques: ["Snap ML", "SnapRandomForestRegressor", "AutoAI Pipeline"],
      metric: null,
      script: "experiments/ibm-experiments/_P5 - Snap random forest regressor_ ....ipynb",
      readme: "experiments/ibm-experiments/README.md",
      models: ["SnapRandomForestRegressor"],
      dataset: "Regression Benchmark",
      details: "Model trained in AutoAI with automated mathematical transformations."
    },
    {
      id: 102,
      title: "Databricks AutoML — DeepAR Forecast",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "external",
      description: "Notebook generated by Databricks AutoML using the DeepAR probabilistic model in PyTorch/GluonTS.",
      techniques: ["DeepAR", "PyTorch", "Probabilistic Forecasting", "Databricks AutoML"],
      metric: null,
      script: "experiments/databricks-forecast/26-01-30-12_17-DeepAR-2f0e47487cb46323278f1a345f799b99.ipynb",
      readme: "experiments/databricks-forecast/README.md",
      models: ["DeepAR (RNN)"],
      dataset: "quantity_sales_transactions",
      details: "Autoregressive recurrent neural network for demand probability distributions."
    },
    {
      id: 103,
      title: "Databricks AutoML — Prophet Forecast",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "external",
      description: "Notebook generated by Databricks AutoML with tuning via Hyperopt and SparkTrials.",
      techniques: ["Prophet", "Hyperopt", "SparkTrials", "Databricks AutoML"],
      metric: null,
      script: "experiments/databricks-forecast/26-01-30-12_17-Prophet-19d52e499b042d483dc0d841414c98e2.ipynb",
      readme: "experiments/databricks-forecast/README.md",
      models: ["Prophet (Databricks)"],
      dataset: "quantity_sales_transactions",
      details: "Distributed hyperparameter tuning on Apache Spark with MLflow tracking."
    },
    {
      id: 104,
      title: "Databricks Local Equivalent",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "completed",
      description: "Prophet+Optuna/SARIMA/ETS on synthetic sales. Open-source equivalent of Databricks.",
      techniques: ["Prophet", "Optuna", "SARIMA", "ETS"],
      metric: {"label": "sMAPE", "value": "5.66%", "percent": 94},
      script: "experiments/time_series/databricks-forecast-local-equivalent.ipynb",
      readme: "experiments/databricks-forecast/README.md",
      models: ["Prophet+Optuna", "SARIMA", "ETS"],
      dataset: "Synthetic Sales",
      details: "Prophet+Optuna sMAPE 5.66% (+11.4% vs baseline). Replaces Databricks AutoML."
    },
    {
      id: 105,
      title: "Watsonx Local Equivalent (Forecast)",
      category: "ibm",
      categoryLabel: "IBM Watsonx / Databricks",
      status: "completed",
      description: "Prophet+Optuna/SARIMA/ETS on Electric Production. Open-source equivalent of Watsonx.",
      techniques: ["Prophet", "SARIMA", "ETS", "Optuna"],
      metric: {"label": "MAPE", "value": "3.90%", "percent": 96},
      script: "experiments/time_series/ibm-watsonx-local-timeseries.ipynb",
      readme: "experiments/ibm-experiments/README.md",
      models: ["Prophet+Optuna", "SARIMA", "ETS", "Naive"],
      dataset: "Electric Production",
      details: "Prophet+Optuna RMSE 3.5583. SARIMA ties at MAPE 3.90% while being 24x faster."
    },
    {
      id: 110,
      title: "MLOps: FastAPI Serving + MLflow Registry",
      category: "mlops",
      categoryLabel: "MLOps Production",
      status: "completed",
      description: "Complete production pipeline: serving, PSI drift, automatic retrain, live dashboard.",
      techniques: ["FastAPI", "MLflow Registry", "PSI Drift", "Auto-Retrain", "Precomputed Forecast"],
      metric: {"label": "Warm latency", "value": "~90ms", "percent": 99},
      script: "mlops/serve.py",
      readme: "mlops/README_mlops.md",
      models: ["SalesForecasterV2 (LightGBM)"],
      dataset: "Sales Forecast V2.2",
      details: "Precompute 12 weeks -> numpy lookup ~80-100ms. 1.6k x speedup. Cooldown 1800s."
    },
    {
      id: 94,
      title: "Ordinal vs Nominal (Wine Quality)",
      category: "regression",
      categoryLabel: "Tabular Regression",
      status: "completed",
      description: "Nominal LogReg/RF vs ordinal LogisticAT/IT (mord). RF wins on acc; the ordinal models tie on acc+-1.",
      techniques: ["Ordinal Regression", "mord", "Random Forest", "Kappa", "MAE"],
      metric: {"label": "RF Acc", "value": "0.660", "percent": 66},
      script: "experiments/ordinal_classification/ordinal_classification.ipynb",
      readme: "experiments/ordinal_classification/README.md",
      models: ["RandomForest", "LogisticRegression", "LogisticAT", "LogisticIT"],
      dataset: "Wine Quality Red (1599x12, 6 classes)",
      details: "RF 0.66/MAE 0.36/Kappa 0.45. Ordinal ~0.59 acc but 0.9775 acc+-1. Rare classes (3,8) collapse."
    },
    {
      id: 602,
      title: "Generative DeepAR — Scenarios and Probabilities",
      category: "timeseries",
      categoryLabel: "Time Series",
      status: "completed",
      description: "DeepAR as a generative model: 500 sampled trajectories, scenarios and event probabilities.",
      techniques: ["DeepAR", "GluonTS", "Generative Forecasting", "Scenarios", "CRPS"],
      metric: {"label": "Trajectories", "value": "500", "percent": 90},
      script: "experiments/time_series/deepar-generative/deepar-generative-futures.ipynb",
      readme: "experiments/time_series/README.md",
      models: ["DeepAR (GluonTS/PyTorch)"],
      dataset: "Benchmark TS (CO2/Nile/Sunspots/Synthetic)",
      details: "Extends the probabilistic DeepAR: massive sampling, scenario bands and P(event) for decision-making."
    }
  ];

  /* -------------------------------------------------------
     Category metadata
     ------------------------------------------------------- */
  const categories = {
    'all': { label: 'All', icon: '\u{1F3AF}', color: '#818cf8' },
    'rl': { label: 'Reinforcement Learning', icon: '\u{1F9BE}', color: '#ef4444' },
    'nlp-sentiment': { label: 'NLP - Sentiment', icon: '\u{1F4AC}', color: '#f472b6' },
    'nlp-class': { label: 'NLP - Classification', icon: '\u{1F4F0}', color: '#fb923c' },
    'hierarchical': { label: 'Hierarchical', icon: '\u{1F333}', color: '#c084fc' },
    'feature-selection': { label: 'Feature Selection', icon: '\u{1F9EC}', color: '#e879f9' },
    'cv': { label: 'Computer Vision', icon: '\u{1F441}', color: '#38bdf8' },
    'recsys': { label: 'Recommender Systems', icon: '\u{1F3AC}', color: '#22d3ee' },
    'timeseries': { label: 'Time Series', icon: '\u{1F4C8}', color: '#34d399' },
    'anomaly': { label: 'Anomalies', icon: '\u{1F50D}', color: '#a78bfa' },
    'clustering': { label: 'Clustering', icon: '\u{1F52E}', color: '#c4b5fd' },
    'regression': { label: 'Tabular Regression', icon: '\u{1F4CA}', color: '#f87171' },
    'ibm': { label: 'IBM Watsonx / Databricks', icon: '\u2601\uFE0F', color: '#fbbf24' },
    'mlops': { label: 'MLOps Production', icon: '\u{1F680}', color: '#4ade80' },
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
        <div class="sidebar-section-title">Categories</div>${navItems}
      </nav>
      <div class="sidebar-stats">
        <div class="sidebar-stats-grid">
          <div class="sidebar-stat"><div class="value">${stats.total}</div><div class="label">Experiments</div></div>
          <div class="sidebar-stat"><div class="value">${stats.completed}</div><div class="label">Completed</div></div>
          <div class="sidebar-stat"><div class="value">${stats.categoriesCount}</div><div class="label">Categories</div></div>
          <div class="sidebar-stat"><div class="value">${stats.techniques}</div><div class="label">Techniques</div></div>
        </div>
      </div>`;
  }

  function renderTopbar() {
    const catLabel = categories[state.activeCategory]?.label || 'All';
    const filtered = getFilteredExperiments();
    return `
      <button class="mobile-menu-btn" onclick="App.toggleSidebar()">\u2630</button>
      <div class="topbar-title">${catLabel} <span>${filtered.length} experiments</span></div>
      <div class="search-container">
        <span class="search-icon">\u{1F50E}</span>
        <input type="text" class="search-input" placeholder="Search experiments, techniques..."
          value="${state.searchQuery}" oninput="App.setSearch(this.value)" />
      </div>
      <div class="topbar-filters">
        <button class="filter-btn ${state.statusFilter==='all'?'active':''}" onclick="App.setStatus('all')">All</button>
        <button class="filter-btn ${state.statusFilter==='completed'?'active':''}" onclick="App.setStatus('completed')">Completed</button>
        <button class="filter-btn ${state.statusFilter==='partial'?'active':''}" onclick="App.setStatus('partial')">Partial</button>
        <button class="filter-btn ${state.statusFilter==='external'?'active':''}" onclick="App.setStatus('external')">External</button>
      </div>`;
  }

  function renderOverview() {
    const s = getStats();
    const pct = Math.round((s.completed / s.total) * 100);
    return `
      <div class="overview-section">
        <div class="overview-header">
          <div>
            <h1>Repository of <span class="gradient-text">MLOps Experiments</span></h1>
            <p>A learning journey through ML, NLP, CV, Time Series, RecSys, MLOps and more</p>
          </div>
        </div>
        <div class="stats-row">
          <div class="stat-card animate-in animate-in-1">
            <div class="stat-icon" style="background:rgba(129,140,248,0.15);color:#818cf8;">\u{1F9EA}</div>
            <div class="stat-value">${s.total}</div><div class="stat-label">Experiments</div>
            <div class="stat-change positive">${pct}% completed</div>
          </div>
          <div class="stat-card animate-in animate-in-2">
            <div class="stat-icon" style="background:rgba(52,211,153,0.15);color:#34d399;">\u2714</div>
            <div class="stat-value">${s.completed}</div><div class="stat-label">Completed</div>
          </div>
          <div class="stat-card animate-in animate-in-3">
            <div class="stat-icon" style="background:rgba(244,114,182,0.15);color:#f472b6;">\u{1F4E6}</div>
            <div class="stat-value">${s.categoriesCount}</div><div class="stat-label">Categories</div>
          </div>
          <div class="stat-card animate-in animate-in-4">
            <div class="stat-icon" style="background:rgba(251,191,36,0.15);color:#fbbf24;">\u{1F527}</div>
            <div class="stat-value">${s.techniques}</div><div class="stat-label">Unique Techniques</div>
          </div>
          <div class="stat-card animate-in animate-in-5">
            <div class="stat-icon" style="background:rgba(167,139,250,0.15);color:#a78bfa;">\u{1F3C6}</div>
            <div class="stat-value">98.57%</div><div class="stat-label">Best F1 (NLP)</div>
          </div>
        </div>
      </div>`;
  }

  function renderCharts() {
    return `
      <div class="charts-section">
        <div class="chart-card animate-in animate-in-1">
          <h3>Distribution by Category</h3><p class="chart-subtitle">Number of experiments per area</p>
          <div class="chart-canvas-wrap"><canvas id="chart-categories"></canvas></div>
        </div>
        <div class="chart-card animate-in animate-in-2">
          <h3>Experiment Status</h3><p class="chart-subtitle">Overall progress</p>
          <div class="chart-canvas-wrap"><canvas id="chart-status"></canvas></div>
        </div>
        <div class="chart-card animate-in animate-in-3">
          <h3>Senti-Pred Evolution</h3><p class="chart-subtitle">Accuracy of the pipelines</p>
          <div class="chart-canvas-wrap"><canvas id="chart-evolution"></canvas></div>
        </div>
        <div class="chart-card animate-in animate-in-4">
          <h3>Techniques Radar</h3><p class="chart-subtitle">Distribution by approach type</p>
          <div class="chart-canvas-wrap"><canvas id="chart-radar"></canvas></div>
        </div>
      </div>`;
  }

  function renderExperimentCard(exp, index) {
    const sl = { completed:'Completed', partial:'Partial', blocked:'Blocked', external:'External' };
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
      <h3>No experiments found</h3><p>Try adjusting the filters</p></div>`;
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
          <h2>${c.label}</h2><span class="category-count">${ce.length} experiments</span>
        </div>
        <div class="experiments-grid">${ce.map((e,i) => renderExperimentCard(e,i)).join('')}</div>
      </div>`;
    }).join('');
  }

  function renderDetailOverlay(exp) {
    if (!exp) return '';
    const sl = { completed:'Completed', partial:'Partial', blocked:'Blocked', external:'External' };
    const models = exp.models.map(m => `<div class="detail-model-item"><span class="model-dot"></span>${m}</div>`).join('');
    const tags = exp.techniques.map(t => `<span class="tag highlight">${t}</span>`).join('');
    const mh = exp.metric ? `<div class="detail-info-item"><div class="info-label">${exp.metric.label}</div>
      <div class="info-value mono">${exp.metric.value}</div></div>` : '';
    const readmeBtn = exp.readme
      ? `<a class="detail-readme-link" href="${GH_BASE}${exp.readme}" target="_blank" rel="noopener">
           \u{1F4D6} View README on GitHub \u2197</a>`
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
        <div class="detail-section"><h4>Information</h4>
          <div class="detail-info-grid">
            <div class="detail-info-item"><div class="info-label">Category</div><div class="info-value">${exp.categoryLabel}</div></div>
            <div class="detail-info-item"><div class="info-label">Dataset</div><div class="info-value">${exp.dataset}</div></div>
            ${mh}
            <div class="detail-info-item"><div class="info-label">Status</div><div class="info-value">${sl[exp.status]||exp.status}</div></div>
          </div>
        </div>
        <div class="detail-section"><h4>Techniques</h4><div class="card-tags">${tags}</div></div>
        <div class="detail-section"><h4>Models</h4><div class="detail-models-list">${models}</div></div>
        <div class="detail-section"><h4>Script / Path</h4><div class="detail-script-path">${exp.script}</div></div>
        <div class="detail-section"><h4>Details</h4>
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
        {label:'Completed',count:sc.completed||0,color:Charts.COLORS.completed},
        {label:'Partial',count:sc.partial||0,color:Charts.COLORS.partial},
        {label:'External',count:sc.external||0,color:Charts.COLORS.external},
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
