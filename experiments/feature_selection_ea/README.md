# Feature Selection Evolucionaria: GAAP (NSGA-II) e MO-DE vs Metodos Classicos

> **Area:** Feature Selection / Otimizacao Multiobjetivo
> **Tarefa:** Regressao (California Housing) e Classificacao multiclasse (Twitter)
> **Metricas:** R2 (regressao) e F1-macro (classificacao) — CV e holdout
> **Status:** Concluido
> **Datasets:** California Housing (20.640 x 44 poly) e Twitter Entity Sentiment (5.000 x 400 TF-IDF)

---

## 1. Resumo

Este experimento compara algoritmos evolucionarios de selecao de features —
**GAAP** (GA com NSGA-II) e **MO-DE** (Differential Evolution multiobjetivo),
ambos implementados em **DEAP** — contra metodos classicos (SelectKBest,
importancia de Random Forest e Boruta). A avaliacao e feita pela curva
**score CV x numero de features** em dois dominios distintos: regressao
(California Housing com features polinomiais interativas) e classificacao
(TF-IDF de tweets). Os evolucionarios mostraram vantagem significativa quando
existe estrutura interativa entre features (California), atingindo ~0,69 de R2
com metade das features do baseline; no dominio bag-of-words (Twitter), os
classicos continuam superiores. O melhor subset de cada metodo e validado no
holdout (teste).

## 2. Contexto e Objetivos

A selecao de features e um problema combinatorio NP-difícil que afeta
diretamente custo de inferencia, interpretabilidade e generalizacao. Metodos
classicos baseados em ranking univariado ou importancia ignoram
**complementaridade entre features** (ex.: pares cujo poder preditivo so
aparece juntos). As questoes de pesquisa:

- (RQ1) Algoritmos evolucionarios encontram subsets de **menor cardinalidade**
  com score competitivo ao do modelo com todas as features?
- (RQ2) O ganho depende da **estrutura do espaco de features** (interacoes vs.
  features quase independentes)?

## 3. Fundamentacao Teorica (curta)

- **Selecao de features** como busca por um subset $S$ que maximiza uma
  metrica de validacao $f(S)$ sob um orçamento de cardinalidade (filter,
  wrapper, embedded).
- **NSGA-II (Deb et al., 2002)**: elitismo por dominancia de Pareto em duas
  frentes: minimizar $(1 - \text{score})$ e minimizar $|S|$. Representacao
  binaria (gene = feature presente).
- **MO-DE**: evolucao de vetores reais em $[0,1]^n$ com mutacao
  $v = v_{r1} + F(v_{r2} - v_{r3})$ e crossover binomial; o subset e obtido
  por limiar $\ge 0.5$; manutencao de frente nao-dominada.
- **Baselines**: ranking por estatistica univariada (f_regression/f_classif),
  feature importance (RF) e sombras/importancia (Boruta); curva top-k
  monotonicamente crescente.

## 4. Metodologia

### 4.1 Dados

| Dataset | Shape | Tarefa | Metrica | Split |
|---|---|---|---|---|
| California Housing | 20.640 x 44 (poly grau 2, log1p) | Regressao | R2 | 80/20 (seed 42) |
| Twitter Entity Sentiment | 5.000 x 400 (TF-IDF 1-2g) | Classificacao 4 classes | F1-macro | 80/20 (seed 42) |

### 4.2 Pre-processamento

- California: log1p em variaveis assimetricas + `PolynomialFeatures(degree=2)`
  sobre dados padronizados (8 -> 44 features interativas).
- Twitter: `TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),
  max_features=400, min_df=2)` com limpeza de URLs/mentions.

### 4.3 Metodos comparados

| Metodo | Tipo | Configuracao |
|---|---|---|
| SelectKBest | filtro univariado | f_regression / f_classif, curva top-k |
| RandomForest | embedded | importance ranking, curva top-k |
| Boruta | wrapper/sombra | perc=90, n_estimators=40 (subamostra) |
| GAAP (NSGA-II) | evolucionario | pop=24, ngen=35 (cal) / 18x25 (tw); cxTwoPoint, mutFlipBit, selNSGA2 |
| MO-DE | evolucionario | pop=30, ngen=40 (cal) / 22x30 (tw); cr=0.5, fw=0.7 |

Modelo base do avaliador: `Ridge(alpha=1.0)` + `StandardScaler` (regressao) e
`LogisticRegression(C=1.0, class_weight='balanced')` (classificacao), com CV
interno de 3 folds.

### 4.4 Avaliacao

- Protocolo: CV interno (3 folds) para construir as curvas score x features;
  o **melhor subset** de cada metodo (maior CV) e re-treinado e avaliado no
  **holdout** (test_score).
- Seeds fixas (42) para reproducibilidade; execucao em CPU (~4 min cal +
  ~2,5 min twitter).

### 4.5 Reproducao

```bash
python feature_selection_ea.py             # pipeline completo
python feature_selection_ea.py --quick     # config reduzida
python build_notebook.py                   # gera/executa o notebook com outputs
```

Artefatos em `outputs/` (`summary_*.csv`, `curves_*.csv`, `*.png`).

## 5. Resultados

### 5.1 Melhor ponto por metodo (CV) e holdout

**California Housing (R2; full = 0.7101)**

| Metodo | best_cv | best_feats | test_score |
|---|---|---|---|
| Boruta | 0.7101 | 44 | 0.7025 |
| SelectKBest | 0.7101 | 44 | 0.7025 |
| RandomForest | 0.7101 | 44 | 0.7025 |
| GAAP (NSGA-II) | 0.6941 | **23** | 0.6807 |
| MO-DE | 0.6831 | **20** | 0.6696 |

**Twitter (F1-macro; full = 0.469)**

| Metodo | best_cv | best_feats | test_score |
|---|---|---|---|
| SelectKBest | 0.4965 | 286 | 0.4610 |
| RandomForest | 0.4722 | 343 | 0.4589 |
| Boruta | 0.4690 | 400 | 0.4711 |
| GAAP (NSGA-II) | 0.4547 | 182 | 0.4394 |
| MO-DE | 0.4501 | 198 | 0.4405 |

### 5.2 Comparacao em orcamento igual de features (California, R2)

| k features | GAAP | MO-DE | RandomForest | SelectKBest |
|---|---|---|---|---|
| ~13 | 0.678 | 0.665 | 0.662 | 0.592 |
| ~19-23 | 0.694 | 0.683 | 0.687 | 0.582 |
| 44 (full) | — | — | — | 0.710 |

### 5.3 Comparacao em orcamento igual de features (Twitter, F1-macro)

| k features | GAAP | MO-DE | RandomForest | SelectKBest |
|---|---|---|---|---|
| 172-182 | 0.443-0.455 | 0.429-0.440 | 0.461 | 0.478 |
| 229-286 | — | — | 0.471 | 0.490-0.497 |

## 6. Discussao

- **California (features interativas): os evolucionarios vencem.** GAAP alcanca
  0.6941 com 23 features (vs 0.7101 com 44). Em orcamentos iguais o GAAP
  domina SelectKBest (0.678 vs 0.592 em k=13). Rankings top-k falham porque
  ignoram interacoes do tipo `MedInc x Latitude`; Boruta colapsa em k medio
  (0.098 em 19 features).
- **Twitter (bag-of-words): classicos ja davam conta.** Features TF-IDF sao
  quase independentes, sem interacao a ser descoberta. SelectKBest em k=229
  (0.490) supera o melhor ponto do GAAP (0.4547 em 182). Note ainda que
  SelectKBest em k=286 (0.4965) supera o score com **todas** as 400 features
  (0.469) — mais features so degradam a LogReg neste dominio.
- **Custo**: GA/DE custam 60-93 s por execucao no Twitter contra ~7 s dos
  baselines, sem ganho em dominios sem interacao.
- **Limitaoes**: resultados de uma seed; NSGA-II/DE sao estocasticos; o
  avaliador e simples (Ridge/LogReg), portanto a conclusao vale para esse
  protocolo.

## 7. Conclusoes e Recomendacoes

1. EA entrega **frente de Pareto** (score x cardinalidade) e so e
   vantajoso quando ha **estrutura interativa** entre features (caso
   California/poly).
2. Em representacoes bag-of-words (TF-IDF), o ranking univariado
   (SelectKBest) ou importancia (RF) ja alcanca o mesmo ou melhor com menor
   custo.
3. Regra pratica: use EA quando features derivadas/interativas dominam o
   espaco; caso contrario, comece por SelectKBest/RF.

## 8. Referencias e Arquivos

- Implementacao: `feature_selection_ea.py`
- Notebook executado: `feature_selection_ea.ipynb` (builder: `build_notebook.py`)
- Resultados: `outputs/` (`summary_cal.csv`, `summary_twitter.csv`, `curves_*.csv`, `*.png`)
- DEAP: Fortin et al., 2012 — DEAP: Evolutionary Algorithms Made Easy.
- Deb et al., 2002 — A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
