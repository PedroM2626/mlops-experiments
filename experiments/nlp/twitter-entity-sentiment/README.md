# Twitter Entity Sentiment Analysis 🐦

This directory centralizes all experiments, notebooks and pipelines focused on the Kaggle **[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)** dataset.

The dataset contains about 74.000 training tweets focused primarily on topics about entities, brands and video games (e.g. Microsoft, Verizon, Borderlands), labeled with 4 sentiments: `Positive`, `Negative`, `Neutral`, `Irrelevant`.

## Directory Structure

- [`NLP-twitter-methods-comparasion.ipynb`](./NLP-twitter-methods-comparasion.ipynb): Exploratory notebook comparing traditional methods.
- [`twitter-sentiment-analysis.ipynb`](./twitter-sentiment-analysis.ipynb): Classical baseline approach (also referenced as **Pipeline B**).
- [`senti-pred_pipeline.ipynb`](./senti-pred_pipeline.ipynb): Notebook of **Pipeline A** (focused on strong/conservative preprocessing).
- [`logistic-regression-multiclass.ipynb`](./logistic-regression-multiclass.ipynb): Analysis with logistic regression.
- [`feature-engineering-nlp.ipynb`](./feature-engineering-nlp.ipynb): Extraction of *features* and NLP aimed at Twitter data.

### Subprojects

1. **[`senti-pred-variations/`](./senti-pred-variations/)**
   - Contains *remake 2* (**Pipeline C**) focused on the power of the vectorizer (100k features, up to 4-grams).
   
2. **[`pipelines_abc_comparison/`](./pipelines_abc_comparison/)**
   - Contains the rigorous orchestration comparing ablations across Pipelines A, B and C, measuring external validity and cross-validation and tracking metrics in **MLflow**. Read the README of that subfolder for the full diagnosis (E1 to E10) and the recipe for the "State of the Art" for this problem.
