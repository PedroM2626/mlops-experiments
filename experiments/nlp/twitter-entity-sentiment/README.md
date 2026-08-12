# Twitter Entity Sentiment Analysis 🐦

Este diretório centraliza todos os experimentos, notebooks e pipelines focados no dataset **[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)** do Kaggle.

O dataset contém cerca de 74.000 tweets de treinamento focados primariamente em tópicos sobre entidades, marcas e videogames (ex: Microsoft, Verizon, Borderlands), rotulados em 4 sentimentos: `Positive`, `Negative`, `Neutral`, `Irrelevant`.

## Estrutura do Diretório

- [`NLP-twitter-methods-comparasion.ipynb`](file:///d:/mlops-experiments/experiments/nlp/twitter-entity-sentiment/NLP-twitter-methods-comparasion.ipynb): Notebook exploratório comparando métodos tradicionais.
- [`twitter-sentiment-analysis.ipynb`](file:///d:/mlops-experiments/experiments/nlp/twitter-entity-sentiment/twitter-sentiment-analysis.ipynb): Abordagem clássica de baseline (também referenciada como **Pipeline B**).
- [`senti-pred_pipeline.ipynb`](file:///d:/mlops-experiments/experiments/nlp/twitter-entity-sentiment/senti-pred_pipeline.ipynb): Notebook da **Pipeline A** (focado num pré-processamento forte/conservador).
- [`logistic-regression-multiclass.ipynb`](file:///d:/mlops-experiments/experiments/nlp/twitter-entity-sentiment/logistic-regression-multiclass.ipynb): Análise com regressão logística.
- [`feature-engineering-nlp.ipynb`](file:///d:/mlops-experiments/experiments/nlp/twitter-entity-sentiment/feature-engineering-nlp.ipynb): Extração de *features* e NLP voltado aos dados do Twitter.
- [`run_twitter_mamba.py`](file:///d:/mlops-experiments/experiments/nlp/twitter-entity-sentiment/run_twitter_mamba.py): Script de inferência/treinamento (experimental).

### Subprojetos

1. **[`senti-pred-variations/`](file:///d:/mlops-experiments/experiments/nlp/twitter-entity-sentiment/senti-pred-variations/)**
   - Contém o *remake 2* (**Pipeline C**) com foco no poder do vetorizador (100k features, até 4-gramas).
   
2. **[`pipelines_abc_comparison/`](file:///d:/mlops-experiments/experiments/nlp/twitter-entity-sentiment/pipelines_abc_comparison/)**
   - Contém a orquestração rigorosa comparando ablações entre as Pipelines A, B e C, medindo validade externa, cross-validation e rastreando métricas no **MLflow**. Leia o README dessa subpasta para o diagnóstico completo (E1 a E10) e a receita do "Estado da Arte" para esse problema.
