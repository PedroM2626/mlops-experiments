# NLP em Regressão — Previsão de Pontuação de Vinhos (Kaggle)

> **Área:** NLP + Regressão (tabular-texto)
> **Tarefa:** Regressão contínua (rating 80–100)
> **Métrica principal:** MAE, R²
> **Status:** Concluído
> **Datasets:** Kaggle Wine Reviews (descrição textual de especialistas)

## 1. Resumo

Diferentemente da classificação clássica de sentimento (Positivo/Negativo), este experimento explora **NLP para regressão contínua**: prever a nota exata de um vinho (escala 80–100) a partir exclusivamente da descrição textual de especialistas. A partir de ~15.000 features geradas por TF-IDF sobre o texto, o modelo **Ridge Regression (linear)** atingiu a melhor performance (MAE 1.33, R² 0.69) treinando em segundos, enquanto o **LightGBM** (gradient boosting em árvores, estado da arte em tabelas densas) sofreu dramaticamente com a esparsidade do texto (MAE 1.47, R² 0.63). O experimento fornece uma prova empírica de como algoritmos lineares prosperam em espaços altamente dimensionais e esparsos.

## 2. Contexto e Objetivos

O estudo foi motivado pela pergunta: qual é o comportamento de algoritmos lineares vs. baseados em árvores quando a dimensionalidade é extremamente alta e a matriz de features é predominantemente esparsa? Nos experimentos de classificação deste repositório (ver `../nlp/README.md`) observou-se que modelos lineares (SVM, LogReg) prosperam em TF-IDF com 70k features; aqui o objetivo foi transferir esse conhecimento para uma **tarefa de regressão contínua** com apenas texto como preditor.

- **Hipótese 1:** Ridge (linear, regularizado) deve superar LightGBM (GBM) em espaços esparsos de alta dimensionalidade.
- **Hipótese 2:** LightGBM, apesar de "estado da arte" em dados tabulares densos, deve degradar pela impossibilidade de encontrar *splits* explorando apenas subespaços esparsos.

## 3. Fundamentação Teórica (curta)

- **TF-IDF** — representação esparsa onde cada termo é uma dimensão; pesos escalam com a frequência local e são amortecidos pela frequência no corpus (IDF).
- **Ridge Regression** — Regressão Linear com regularização L2; em espaços de alta dimensionalidade esparsa, a relativa tem um problema bem-comportado e convergê em poucos segundos.
- **LightGBM** — Gradient Boosting sobre árvores; brilha com features densas/tabulares, mas baseia decisões em *splits* binárias de features — em espaço esparso, bons cortes são raros e o custo de busca cresce muito.
- **MAE / R²** — erro absoluto médio e coeficiente de determinação usados como métricas principais.

## 4. Metodologia

### 4.1 Dados
- Fonte: Kaggle (Wine Reviews — descrições de especialistas).
- Target: pontuação do vinho (escala contínua, ~80–100).
- Features: vetorização **TF-IDF** do texto, com um vocabulário de aproximadamente **15.000 features**.

> Nota: a descrição do experimento no README raiz usa MAE 1.33 e R² 0.69 para Ridge e MAE 1.47 / R² 0.63 para LightGBM (execução consolidada). Consulte os notebooks/MLflow para o valor exato (o artefato `mlruns/` registra a execução com `tfidf_vectorizer.pkl`).

### 4.2 Pré-processamento
- Limpeza de texto e vetorização TF-IDF.
- Saldo do vetorizador em `tfidf_vectorizer.pkl` (artefato no nível da pasta e via MLflow).

### 4.3 Métodos comparados

| Paradigma | Modelo | Configuração |
|---|---|---|
| Linear | **Ridge Regression** | TF-IDF ~15k features, regularização L2 |
| Árvores | **LightGBM Regressor** | mesmo TF-IDF, busca de splits esparsos |

### 4.4 Avaliação
- Métricas: **MAE** (erro absoluto médio) e **R²**.
- Rastreamento: MLflow (execuções sob `mlruns/`, com artefatoa de vetorizador e imagens `target_distribution.png`, `real_vs_predicted.png`).

### 4.5 Reprodução
- Notebook principal: `nlp_regression_wine.ipynb` (pasta atual).
- Vetorizador pré-treinado: `tfidf_vectorizer.pkl`.
- Artefatos MLflow: `mlruns/1/<run_id>/artifacts/` (modelo `model/MlModel`, png de diagnóstico).

## 5. Resultados

| Método | MAE | R² | Observações |
|---|---|---|---|
| **Ridge Regression** | **1.13** | **0.69** | Treino em poucos segundos — melhor performance |
| **LightGBM** | **1.47** | **0.63** | Sofre com esparsidade; demanda muito processamento para *splits* |

*(Valores line referidos no README raiz; confirmar variação exata no notebook/MLflow caso exista diferença entre execuções — na raiz também aparece MAE 1.33/R² 0.69 e 1.47/0.63.)*

Interpretação:
- A **Ridge** linear aproveita a geometria do espaço esparso de 15.000 termos, encontrando solução regularizada rapidamente.
- O **LightGBM** tenta encontrar *splits* em um oceano de zeros; sofrendo forte redução de performance (R² 0.63 vs. 0.69).

## 6. Discussão

Este resultado confirma o padrão observado nos experimentos de classificação NLP do repositório: **modelos lineares prosperam em representações esparsas de alta dimensionalidade**, enquanto árvores precisam de features densas/informativas para encontrar boas bordas de decisão. A esparsidão espacial do texto (a maioria dos termos ocorre em poucos documentos) faz com que os *splits* do LightGBM percam a vista de sinal; a regularização L2 do Ridge, por sua vez, conseguirá distribuir peso entre os muitos termos discriminativos sem overfitting.

Caveat: não foram reportadas seeds e hardware nos artefatos; espera-se que a diferença Ridge vs LightGBM seja consistente (linear vence esparso), mas os valores exatos podem variar com a amostra/limpeza de texto.

## 7. Conclusões e Recomendações

- Para **regressão sobre texto** (feature descriptiva), prefere modelagens **lineares regulares (Ridge/Lasso)** sobre Tree-based — representação esparsa inicial.
- **LightGBM** deve ser limitado a dados tabulares densos; se necessário, reduzir a dimensionalidade (SVD/truncated SVD) ou usar embeddings densos para tornar o espaço amigável a árvores.
- **Recomendação:** usar a pipeline TF-IDF → Ridge (MAE 1.33/R² 0.69) como baseline de regressão textual; experimentar com `truncatedSVD` antes de árvores para recuperar performance, se necessário.

## 8. Referências e Arquivos

- `nlp_regression_wine.ipynb` — notebook principal.
- `tfidf_vectorizer.pkl` — vetorizador pré-treinado (pasta local + MLflow artifact).
- `mlruns/` — execuções registradas no MLflow, com `model/ModelML` e imagens de diagnóstico (`target_distribution.png`, `real_vs_predicted.png`).
- Notebooks relacionados (classificação NLP): `../nlp/README.md`.