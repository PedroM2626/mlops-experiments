# Regressao Tabular: Feature Engineering e Predicao de Precos

> **Area:** Regressao tabular / AutoML local
> **Tarefa:** Predicao de variavel continua (preco imobiliario, preco de automovel)
> **Metrica principal:** R2, MAE, RMSE
> **Status:** Concluido
> **Datasets:** California Housing (20.640 amostras), price-prediction-multiple-linear-regression (205 amostras)

---

## 1. Resumo

Este grupo reune tres estudos complementares de regressao tabular: (i) um
estudo sistematico de **10 tecnicas de feature engineering** sobre California
Housing com LinearRegression, LightGBM e RandomForest; (ii) a evolucao de um
pipeline de **predicao de precos de automoveis** (v1 -> v2 -> v3) ate o
plateau pratico de R² = 0,9489; e (iii) um equivalente local open-source ao
AutoML da IBM Watsonx. A conclusao transversal: feature engineering tem
**valor assimetrico por modelo** (grande em modelos lineares, marginal em
modelos de arvore) e PCA foi amplamente prejudicial.

## 2. Contexto e Objetivos

- Quantificar o **efeito isolado** de feature engineering em modelos de
  familias distintas (linear vs. arvore) no mesmo dataset.
- Evidenciar a evolucao de um pipeline de preco simples até um pipeline com
  regularizacao, encoding, transformacao do target e tunagem (v1 -> v3).
- Prover uma alternativa 100% open-source ao AutoML da IBM Watsonx rodando
  localmente.

## 3. Fundamentacao Teorica (curta)

- Modelos lineares (OLS, Ridge) capturam apenas relacoes lineares; features
  polinomiais/transformacoes ampliam a expressividade sem trocar de modelo e,
  por invariancia afim, scaling nao altera o resultado do OLS.
- Arvores (RF/LightGBM) aprendem nao-linearidades nativamente e sao
  invariantes a escala; features de domain knowledge (geo) agregam o que
  splits univariados nao derivam.
- PCA maximiza variancia, nao correlacao com o target -- risco de destruir
  informacao direcional em features colineares.

## 4. Metodologia

### 4.1 Feature Engineering Tabular (California Housing)

| Fator | Detalhe |
|---|---|
| Dataset | California Housing, 20.640 x 8 features |
| Tecnicas | Raw, Standardized, MinMax, Polynomial(d=2), Interactions, Log, Binning, PCA(95%), Geo, Combined |
| Modelos | LinearRegression, LightGBM, RandomForest |
| Metricas | R², MAE |
| Seed / HW | 42; Intel i7, 16 GB |

### 4.2 Price Prediction (205 amostras)

- v2: remove `ID`, one-hot de 9 categorias (23 -> 42 features), `log1p` do
  target (assimetria 1,78 -> 0,46), winsorizacao, GridSearchCV em 6 modelos,
  CV 5-folds.
- v3: polynomial (741 feats), ExtraTrees, RF variante, GradientBoosting para
  tentar superar o plateau.

### 4.3 IBM Watsonx local (California Housing, holdout 10%)

- Baselines (Ridge, Lasso, ElasticNet, RF, ET, GB, AdaBoost, SVR, XGB) + FLAML
  (AutoML Bayesiano) + TPOT (AutoML genetico).

### 4.4 Reproducao

```bash
jupyter nbconvert --to notebook --execute feature-engineering-tabular.ipynb
jupyter nbconvert --to notebook --execute price-prediction-multiple-linear-regression.ipynb
jupyter nbconvert --to notebook --execute ibm-watsonx-local-automl.ipynb
```

## 5. Resultados

### 5.1 Feature Engineering - R² por modelo

| Tecnica | LinearRegression | LightGBM | RandomForest |
|---|:--:|:--:|:--:|
| Raw | 0,5758 | 0,8360 | 0,8051 |
| Polynomial (d=2) | 0,6457 | 0,8346 | 0,7968 |
| Log transform | 0,6114 | 0,8360 | 0,8053 |
| Geo features | 0,5945 | **0,8418** | **0,8205** |
| Combined | **0,7112** | 0,8375 | 0,8045 |
| PCA (95%) | 0,4877 | 0,6583 | 0,6422 |

### 5.2 Price Prediction (teste)

| Modelo | R² Teste | MAE | CV R² | Overfit |
|---|---|--:|--:|--:|
| Random Forest (GS) | **0,9489** | 1.043,7 | 0,8897 | 0,0372 |
| XGBoost (GS) | 0,9391 | 1.316,2 | 0,8931 | 0,0576 |
| ElasticNet (GS) | 0,8978 | 1.424,3 | 0,8801 | 0,0194 |
| Ridge (GS) | 0,8968 | 1.461,6 | 0,8823 | 0,0188 |
| Linear Regression | 0,8900 | 1.676,8 | 0,8423 | 0,0478 |

Evolucao v1 -> v2: R² 0,8517 -> 0,9489; MAE -56,7%. v3 (poly/ExtraTrees):
nenhuma abordagem superou o plateau da v2 (limitante = tamanho do dataset).

### 5.3 IBM Watsonx local (holdout)

| Metodo | RMSE | R² | Tempo |
|---|--:|--:|--:|
| XGBoost | 0,4618 | 0,8401 | 1,58s |
| FLAML (CatBoost) | 0,4780 | 0,8286 | 63,9s |
| TPOT | 0,4817 | 0,8260 | 199,1s |
| Extra Trees | 0,4997 | 0,8128 | 1,12s |

## 6. Discussao

- Feature engineering tem valor assimetrico: LinearRegression ganhou +13,5 pp
  (R²) com Combined; LightGBM apenas +0,6 pp (Geo). Scaling nao altera o OLS
  (invarianca afim). PCA perdeu 9-18 pp em todos os modelos.
- Price prediction: log do target + encoding + CV levaram a Linear Regression
  de 0,8517 para 0,8900; ensembles superam lineares em ~5 pp; v3 confirmou
  que o limitante e o tamanho do dataset, nao a complexidade do modelo.
  Residuos do RF normais (Shapiro p=0,09; Jarque-Bera p=0,48).
- AutoML local: XGBoost manual superou FLAML/TPOT por margem pequena; AutoML
  e uma boa baseline automatica.

## 7. Conclusoes e Recomendacoes

1. De o esforco de FE proporcional a familia do modelo: lineares justificam
   horas, arvores minutos (foco em domain knowledge).
2. Para price-prediction, Random Forest (v2) e o modelo final recomendado;
   mais dados seriam o proximo passo.
3. Use PCA com cautela: otimiza variancia, nao correlacao com o alvo.
4. AutoML (FLAML/TPOT) e uma baseline automatica; um XGBoost bem parametrizado
   continua competitivo e muito mais rapido.

## 8. Referencias e Arquivos

- `feature-engineering-tabular.ipynb` -- estudo de FE tabular.
- `price-prediction-multiple-linear-regression.ipynb` -- pipeline v1->v3.
- `ibm-watsonx-local-automl.ipynb` -- equivalente local ao AutoML da Watsonx.
- Estudo cruzado modelo x FE no README raiz (secao Feature Engineering).