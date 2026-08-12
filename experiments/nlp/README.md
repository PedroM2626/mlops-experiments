# Grupo de Experimentos NLP — Sentimento, Tópicos e Representações Textuais

> **Área:** NLP
> **Tarefa:** Classificação (sentimento, tópicos, multi-tarefa) e regressão textual
> **Métrica principal:** F1-Macro / F1-Weighted / Acurácia
> **Status:** Concluído
> **Datasets:** Twitter Entity Sentiment Analysis (73.995 treino / 999 validação), AG News (4 classes), Google `go_emotions`, 20 Newsgroups e dataset de sentimento de textos diversos (7.500 linhas, 15 colunas) como referência de transferência.

## 1. Resumo

Esta pasta reúne a linha de experimentos de NLP do repositório: comparação de paradigms de representação (TF-IDF esparso, embeddings congelados e transformers contextualizados), ensembles hierárquicos (Ensemble Pyramid), otimização de um pipeline de sentimento em redes sociais (Twitter/Facebook/X) e classificação de tópicos em AG News. O resultado principal é que, para a sentiment análise esparsa de alta dimensionalidade (TF-IDF + n-grams), modelos lineares e ensembles randomizados superam transformers fine-tuned na maioria dos cenários (F1 ~0.98), enquanto o fine-tuning regularizado de transformers vence apenas em regimes de baixa amostragem. A engenharia de features (limpeza de texto, n-grams, vocabulário) mostrou-se mais decisiva do que a escolha do modelo.

## 2. Contexto e Objetivos

O projeto surge do questionamento sobre qual representação e qual algoritmo produz o melhor custo-benefício para classificação de texto em produção, sob três pragmáticas pontos de vista:

1. **Custo computacional** — executar em hardware moderado (CPU/GPU de laptop) sem incorrer em semanas de treino.
2. **Acurácia** — atingir estados da arte (F1 ≥ 0.95) em cenários onde os dados de treino são abundantes, e entender quando arquiteturas profundas são necessárias.
3. **Interpretabilidade** — entender onde os erros ocorrem (limpeza do texto vs. vetorização vs. escolha do modelo).

As hipóteses investigadas foram:

- `H1` — Para tweets (textos curtos), representações esparsas TF-IDF com bigramas + SVM linear rivalizam com transformers fine-tuned a uma fração do custo (em segundos vs. horas).
- `H2` — Em regimes de baixa amostragem (N ≤ 1000), modelos clássicos tendem a superar transformers fine-tuned.
- `H3` — Ensembles hierárquicos/meta-ensembles (Ensemble Pyramid) elevam progressivamente o F1 além do melhor modelo individual.
- `H4` — A qualidade do pré-processamento de texto é mais decisiva do que a escolha do algoritmo.

## 3. Fundamentação Teórica (curta)

- **TF-IDF** — *Term Frequency × Inverse Document Frequency*: matriz esparsa onde cada dimensão é um termo do vocabulário; o peso escala com a frequência no documento e é amortecido pela frequência no corpus (IDF). Com `sublinear_tf=True` aplica-se `1 + log(tf)`, atenuando palavras muito repetidas.
- **n-grams** — unigramas/bigramas capturam sentidos de negação (`not good`, `very bad`); n-grams de caracteres (`char_wb` 2–5) capturam padrões morfológicos. Em geral, bigramas em sentimento são discriminativos e frequentes (~5–15% dos documentos), enquanto em tópicos são esparsos (<1%).
- **LinearSVC** — SVM linear com penalidade L2 (parâmetro C); robusto em espaços esparsos de alta dimensionalidade.
- **Transformers** — Self-attention com complexidade quadrática O(N²). **DistilBERT** (66M params): destilado do BERT. **Mamba (SSM, 130M)**: modelos de espaço de estado discretizados, complexidade O(N), mas com overhead fixo das projeções lineares.
- **Ensembles** — Bagging, Voting (Soft/Hard) e Stacking com combinação de modelos. **Épsilon-Greedy** e **Thompson Sampling** usados no controle do RL do Versatile Ensemble Pyramid.
- **MMoE** — *Multi-gate Mixture of Experts*: múltiplas redes especialistas compartilhadas com gates por tarefa; visa mitigar Transferência Negativa, mas é sensível à escala de dados/features.
- **Focal Loss** — variante de entropia cruzada que penaliza dinamicamente amostras difíceis sobre as fáceis; útil para a tarefa desbalanceada.

## 4. Metodologia

### 4.1 Dados

| Experiment | Dataset | Classes | Split |
|---|---|---|---|
| Pipeline A/B (Senti-Pred) | Twitter Entity Sentiment Analysis | 4 (Irrelevant, Negative, Neutral, Positive) | 73.995 treino / 999 validação |
| Ag News | AG News | 4 (World, Sports, Business, Sci/Tech) | 1.000 treino / 200 teste (seed 42) |
| Proportion of Grid | Twitter Entity Sentiment | 4 | 73.768 treino / 999 validação |
| MMoE | Google `go_emotions` | multi-labels (Alegria, Tristeza, Raiva, ...) | até 43.000 amostras |

Hardware: NVIDIA GeForce RTX 4070 Laptop (CUDA 12.1) + Intel i7 / Python 3.8.10. Seeds 42 (numpy/torch).

### 4.2. Pré-processamento

A limpeza de texto evoluiu ao longo da série (detalhamento na §5.3). Variações avaliadas entre Pipeline A (agressivo) e Pipeline B (conservador):

| Componente | Pipeline A | Pipeline B |
|---|---|---|
| Hashtags | Remove `#palavra` inteira (`@\w+\|#\w+`) | Mantém o conteúdo (`#great` → `great`) |
| Pontuação | Remove toda (`[^\w\s]`) | Preserva `!?.,'"` e hífens |
| Números | Remove (`\d+`) | Mantém números (`0-9`) |
| Stopwords | Removidas (F1) / mantidas (fases seguintes) | Mantidas |
| Lematização | WordNet com POS em Fase 1, depois desativada | Não usada |

### 4.3. Métodos comparados

| Experimento | Modelos/ParadDirigidos | Estrutura |
|---|---|---|
| Ensemble Pyramid (6 camadas) | LR, LinearSVC, NB, CNB, Ridge, RF, ET + Bagging/Voting/Stacking | Pirâmide hierárquica de meta-ensembles |
| Versatile Ensemble Pyramid | RL Meta-Learner escolhe nº de modelos e estratégia | AutoML com `--layers` variável |
| Pipeline A / B | Extra Trees, LinearSVC(C=1/10/19), LR, MNB | TF-IDF 15k→70k features |
| Twitter Methods | TF-IDF+LinearSVC, Sentence-BERT frozen, DistilBERT, BiLSTM, TextCNN, Mamba (SSM) | 74j amostras |
| Logística multiclasse | Multinomial(lbfgs), OvR(lbfgs/liblinear/saga), OvO(liblinear) | C ∈ {0.1 … 100} |
| Feature Engineering | TF-IDF vs. hashing trick, word+char n-grams | varias transformed |
| AG News | DistilBERT (fine-tune) vs. TF-IDF+LinearSVC / +ExtraTrees | Low-data 1k |
| MMoE | Single-Task vs. Multi-Abiação MMoE (DistilBERT embeddings vs. TF-IDF) | 4 tags emotion | — |

### 4.4 Avaliação

Métricas: Acurácia, F1-Macro, F1-Weighted (mudadas entre fases), Precision/Recall. Protocolo: holdout treino/validação fixo do dataset; grid search no AG News (max_features 500–5.000); tracking via MLflow + DagsHub (mesma execução agrupada, métricas prefixadas).

### 4.5 Reprodução

- `ag-news-classification.ipynb` — Exp1 AG News (1000 treino / 200 teste, seed 42).
- **Twitter Entity Sentiment Analysis**: Todos os experimentos e pipelines originais (A, B, C) envolvendo este dataset foram centralizados na subpasta `twitter-entity-sentiment/`. Isso inclui `twitter-sentiment-analysis.ipynb`, `senti-pred_pipeline.ipynb`, `logistic-regression-multiclass.ipynb`, `feature-engineering-nlp.ipynb` e `NLP-twitter-methods-comparasion.ipynb`.
- `nlp-multi-task-classification.ipynb` — MMoE multi-task em `go_emotions`.
- `../ensemble_pyramid.py` — Ensemble Pyramid / Versatile Ensemble Pyramid (AutoML CLI).

Padrão de saída de artefatos: `experiments/artifacts/<experimento>_<timestamp>_<sha>/`.

## 5. Resultados

### 5.1. Ensemble Pyramid — 6 Camadas de Ensembles sobre Ensembles

Arquitetura em pirâmide combinando Bagging, Voting e Stacking hierarquicamente:

- **Camada 1**: Base Learners (LR, LinearSVC, NB, CNB, Ridge, RF, ET)
- **Camada 2**: Ensembles dos Base Learners (Bagging + Voting + Stacking)
- **Camada 3**: Ensembles de Ensembles (Stacking + Bagging sobre Stacking + Voting)
- **Camada 4**: Meta-Ensemble Final (Meta Voting Soft + Meta Stacking + Meta Voting Hard)
- **Camada 5**: Meta-Ensemble Intermediário (Meta2 Voting Soft + Meta2 Stacking + Meta2 Voting Hard)
- **Camada 6**: Meta-Ensemble Final Aprimorado (Final Stacking + Final Voting Soft + Final Voting Hard)

Características:
- Manutenção em formato esparso: TF-IDF com 70k features ocupa ~15 MB.
- Classes leves (`PreFittedSoftVoting`, `PreFittedHardVoting`, `MetaStackingLR`) evitam re-treino desnecessário.
- Combina predições probabilísticas de múltiplos níveis hierárquicos.

Resultado principal: **F1-score ~0.98+ na validação**, com ganhos progressivos por camada (ruar, Soti embarcados).

### 5.2. Versatile Ensemble Pyramid (script AutoML personalizável)

Motor de AutoML que usa RL para decidir dinamicamente a arquitetura da pirâmide:

- **Quantidade de Modelos Variável** — o RL Meta-Learner decide quantos e quais modelos por camada (ex.: Camada 1 com 3 modelos, Camada 2 com 2), maximizando diversidade e eficiência.
- **Seleção Estocástica (Thompson Variation)** — o agente mantém um ranking de performance, mas introduz ruído planejado para testar novas sinergias entre as meta-features.

Parâmetros CLI (sem alterar código):

| Parâmetro | Descrição | Exemplo |
|---|---|---|
| `--layers` | Profundidade total da pirâmide | `--layers 15` |
| `--min_models` / `--max_models` | Largura e diversidade por camada | `--min_models 3 --max_models 6` |
| `--epsilon` | Exploração do RL (0.1 focado, 0.5 explorador) | `--epsilon 0.5` |
| `--metric` | Métrica do agente | `f1` ou `accuracy` |
| `--strategy` | Conexão entre camadas | `dense`, `residual`, `simple` |
| `--jitter` | Variação aleatória de hiperparâmetros | `True/False` |
| `--patience` | Camadas sem melhora antes do early stopping | `--patience 3` |
| `--seed` | Reproducibilidade 100% (seeding global) | `--seed 42` |
| `--tfidf_max` / `--tfidf_ngrams` | Customização da extração de features | `--tfidf_max 75000` |

Execução com customização extrema:

```bash
python ../ensemble_pyramid.py --layers 15 --min_models 3 --max_models 6 --strategy dense --jitter True --metric f1 --tfidf_max 75000
```

As configurações são registradas no MLflow automaticamente para comparação entre estratégias de evolução.

### 5.3. Trajetória de evolução do Pipeline A (Senti-Pred)

| Fase | Configuração | Melhor modelo | Acc/F1 |
|---|---|---|---|
| Fase 1 | TF-IDF 15k (unig+bigrama), lematização POS, stopwords removidas | Extra Trees | Acc 0.9750 / F1-macro 0.9744 |
| Fase 2 | TF-IDF 70k (bigramas), sem stopwords, sem lematização | Extra Trees | Acc/F1 0.9820 |
| Fase 3 | TF-IDF 70k + `sublinear_tf=True` + `strip_accents` | Extra Trees | F1 0.9810 (LR subiu p/ 0.9750) |
| Fase 4 | Fase 3 + LinearSVC com C=10 e C=19 | LinearSVC (C=10.0/19.0) | Acc/F1 0.9820 |

**Fase 1 detalhada (15k features, F1-Macro):**

| Modelo | Accuracy | F1-Macro |
|---|---|---|
| Extra Trees | **0.9750** | **0.9744** |
| Linear SVC (C=1.0) | 0.9369 | 0.9362 |
| Logistic Regression | 0.8989 | 0.8960 |
| Multinomial NB | 0.7838 | 0.7753 |

**Fase 2 detalhada (70k features, F1-weighted):**

| Modelo | Acc / F1 |
|---|---|
| **Extra Trees** | **0.9820** |
| Linear SVC (C=1.0) | 0.9800 |
| Logistic Regression | 0.9730 |
| Multinomial NB | 0.9150 |

**Fase 3 detalhada:**

| Modelo | Acc / F1 |
|---|---|
| **Extra Trees** | **0.9810** |
| Linear SVC (C=1.0) | 0.9800 |
| Logistic Regression | 0.9750 (+0.20% com sublinear_tf) |
| Multinomial NB | 0.9140 |

**Fase 4 detalhada (regularização do SVC):**

| Modelo | Acc / F1 |
|---|---|
| **Linear SVC (C=10.0 ou C=19.0)** | **0.9820** |
| Extra Trees | 0.9810 |
| Linear SVC (C=1.0) | 0.9800 |
| Logistic Regression | 0.9750 |
| Multinomial NB | 0.9140 |

### 5.4. Duelo de engenharia: Pipeline A vs. Pipeline B vs. Pipeline C (Senti-Pred-remake2)

- Pipeline B: substitui `#palavra` por `palavra`, conserva pont. `!?.`, hífens e contrações (`don't`), mantém números.
- Pipeline A: remove hashtags por completo, remove toda a pontuação (vira `dont` a partir de `don't`), exclui dígitos.
- Pipeline C (Senti-Pred-remake2): vetorização extrema (TF-IDF 100k, 4-grams), limpeza com
  lematização, stopwords (com `not`/`no` preservados) e expansão de contrações; votação
  LinearSVC(C=0.5, balanced) + LR(C=10, balanced).

Resultado final do duelo (reproduzido nesta execução, seed 42, holdout 1.000):

| Pipeline | Campeã | Acurácia | F1-Macro | F1-Weighted |
|---|---|---|---|---|
| **A** (agressiva) | ExtraTrees | 0.9850 | **0.9845** | 0.9850 |
| **B** (conservadora) | LinearSVC C=19 | 0.9830 | 0.9833 | 0.9830 |
| **C** (remake2) | LinearSVC C=0.5 | 0.9780 | 0.9782 | 0.9780 |

**Conclusão da §5.4:** a engenharia de features (limpeza do texto) foi mais decisiva do que a
escolha do modelo. Ao preservar exclamações, conteúdo de hashtags e contrações idiomáticas, o
Pipeline B gera representações de sentimento mais ricas; a Pipeline A, agressiva, atinge o
**melhor F1-Macro entre as canônicas** com ExtraTrees. O recorde da Pipeline C (~97.8%)
reproduz-se, mas a análise rigorosa de ablações (`pipelines_abc_comparison/README.md`)
mostra que o vetorizador (100k + bigramas) é o ativo mais valioso — não a limpeza: o melhor
F1-Macro do estudo (**0.9857**) surge ao combinar **pré-processamento A + vetorizador C**.
Diferenças < 1 pp entre as três são estatisticamente não-significativas (McNemar, p ≥ 0.33).

**What-ifs principais (detalhes e tabelas em `pipelines_abc_comparison/README.md`):**
- **n-gramas:** remover os bigramas derruba −2.6 a −4.8 pp; a C melhora **+0.33 pp** ao
  trocar 4-grams por bigramas; a B é a mais sensível a N>2 (até −0.61 pp).
- **Vocabulário:** `max_features` 10k→100k na C custa −8.5 pp; 200k rende +0.41 pp; A/B sofrem
  −4 a −4.6 pp se truncados a 10k.
- **Limpeza:** manter hashtags/pontuação/dígitos na A rende ~+0.4 pp cada; manter stopwords na
  C rende +0.30 pp; contrações e conteúdo de hashtags são sinal na C (+0.43/+0.32 pp).
- **Modelo:** o Voting oficial da C é levemente inferior ao LinearSVC C=0.5 isolado;
  `voting='soft'` degrada; `class_weight=balanced` sozinho não explica o ganho da C.
- **Significância:** nenhuma diferença entre pipelines é estatisticamente significativa
  (N=1.000; erros totais 15/17/22).

### Pipelines comparadoras — paths relativos:

- Pipeline A → `senti-pred_pipeline.ipynb`
- Pipeline B → `twitter-sentiment-analysis.ipynb`
- Pipeline C → `pipelines_abc_comparison/` + `../senti-pred-variations/Senti-Pred-remake2/`

### 5.5. Twitter Methods Comparison — Paradigmas de Representação Textual

Notebook: `../NLP-twitter-methods-comparasion.ipynb`. Cinco (seis) paradigmas no dataset completo (73.995 treino / 999 val, 4 classes).

| Modelo | Acurácia | Tempo (s) | Paradigma | Parâmetros |
|---|---|---|---|---|
| **TF-IDF + LinearSVC** | **0.9800** | **4,35** | BoW + SVM linear | ~70M features |
| **DistilBERT** | **0.9710** | 2.421,08 | Transformer | 66M parámetros |
| **Mamba (SSM)** | **TBD** | TBD | State-Space Model (linear head) | 130M |
| TextCNN | 0.9530 | 13,00 | CNN 1D em embeddings | ~2.6M |
| BiLSTM | 0.8900 | 13,26 | LSTM bidirecional | ~1.1M |
| Sentence-BERT | 0.6036 | 33,93 | Transformer congelado + LinearSVC | 22M congelados |

Detalhe: TF-IDF+LinearSVC 0.9800 / 4.35s — acurácia com regularização L2 (C=1), dependendo do vocabulário. Percentagem das tabelas reais:

**TF-IDF + LinearSVC descreve** (weighted 0.98). **DistilBERT** refosa de 0.8529 (30k) para **0.9710** (74k, +11.81 pp; 2.421s, 556× o tempo do TF-IDF). Época 1 do 74k: Loss 0.1962 → Acc 0.9409; Época 2: Loss 0.1003 → Acc 0.9710. **TextCNN** 0.9530/13s (melhor proporção acurácia/tempo entre neurais: 98,5% da performance do DistilBERT em 0,5% do tempo). **BiLSTM** 0.8809/13,26s. **Sentence-BERT** estagnado 0.6036 (ganho de +0,40 pp da subamostra 30k para a completa). **Mamba (SSM)** — em aster? TBD: em textos curtos (~20 tokenários) o ganho assintótico O(N) é suprimido pelo overhead das projeções de 130M pesos; no Windows local cai para fallback sequencial, já que `mamba-ssm` é otimizado apenas via CUDA/Triton.

**Efeito do dataset completo (30k → 74k):**

| Modelo | Acurácia 30k | Acurácia 74k | Ganho (pp) | Tempo 74k (s) |
|---|---|---|---|---|
| TF-IDF + LinearSVC | 0,9800 | 0,9800 | 0,00 | 4,35 |
| DistilBERT | 0,8529 | **0,9710** | **+11,81** | 2.421,08 |
| Mamba (SSM) | - | **TBD** | **-** | TBD |
| TextCNN | 0,7838 | **0,9530** | **+16,92** | 13,00 |
| BiLSTM | 0,7187 | **0,8809** | **+16,22** | 13,26 |
| Sentence-BERT | 0,5996 | 0,6036 | +0,40 | 33,93 |

**Insight central:** o ganho com dataset completo é diretamente proporcional ao nº de parâmetros tweáveis e inversamente proporcional à qualidade da representação inicial. Com Sentence-BERT (0 treino de pesos p) o dataset não resolve (lin mosaic fechada). Com TextCNN/BiLSTM (todos os pesos novos) o ganho cresce +16–17pp.

**Hierarquia de custo-benefício (dataset completo):**

| Paradigo | Acurácia | Tempo (s) | Eficiência (Acc/s) | GPU? |
|---|---|---|---|---|
| **TF-IDF + LinearSVC** | 0,9800 | 4,35 | **0,2253** | Não |
| **TextCNN** | 0,9530 | 13,00 | **0,0733** | Recomendada |
| BiLSTM | 0,8809 | 4,26 | 0,0664 | Recomendada |
| DistilBERT | 0,9710 | 2.421,08 | 0,0004 | Sim |
| Mamba (130M) | TBD | TBD | – | Sim (CUDA estrito) |
| Sentence-BERT | 0,6036 | 33,93 | 0,0178 | Sim |

### 5.6. Logistic Regression: Estratégias Multiclasse

Notebook: `logistic-regression-multiclass.ipynb`. Dataset Twitter Sentiment (73.768 treino/999 val). 5 configurações de `multi_class`, `solver`, `C`.

Estratégias:

| # | Estratégia | `multi_class` | `solver` | Mecanismo |
|---|---|---|---|---|
| 1 | Multinomial | `multinomial` | `lbfgs` | Softmax nativo (probs somam 1) |
| 2 | OvR (lbfgs) | `ovr` | `lbfgs` | K binários, quasi-Newton |
| 4 | OvR (saga) | `ovr` | `saga` | K binários, gradiente estocástico |
| 5 | OvO (liblinear) | (wrap) | `liblinear` | K×(K−1)/2 binários de par, votação |

Resultados por C (Acurácia / F1-weighted):

| Estratégia | C=0.1 | C=1.0 | C=10.0 | C=100.0 | Melhor |
|---|---|---|---|---|---|
| **Multinomial (lbfgs)** | 0,7598 / 0,7516 | 0,9750 / 0,9750 | 0,9820 / 0,9820 | 0,9780 / 0,9780 | 10 (59,93s) |
| OvR (lbfgs) | 0,7137 / 0,6980 | 0,9630 / 0,9630 | 0,9780 / 0,9780 | **0,9800** / 0,9800 | 100 (37,14s) |
| OvR (liblinear) | 0,7137 / 0,6980 | 0,9630 / 0,9630 | 0,9780 / 0,9780 | **0,9790** / 0,9790 | 100 (43,11s) |
| OvR (saga) | 0,7137 / 0,6980 | 0,9630 / 0,9630 | 0,9780 / 0,9780 | **0,9790** / 0,9790 | 100 (41,66s) |
| OvO (liblinear) | 0,6907 / 0,6668 | 0,9530 / 0,9529 | 0,9770 / 0,9770 | **0,9780** / 0,9780 | 100 (6,82s) |

Detalhamento no C=10 (F1 por classe):

| Estratégia | Acurácia | F1-weighted | F1-macro | Tempo (s) | F1 Irrelevant | OGE Negative | F1 Neutral | F1 Positive |
|---|---|---|---|---|---|---|---|---|
| **Multinomial (lbfgs)** | **0,9820** | **0,9820** | **0,9829** | 135,43 | 0,9853 | 0,9857 | 0,9798 | 0,9767 |
| OvR (lbfgs) | 0,9780 | 0,9779 | 0,9777 | 22,39 | 0,9823 | 0,9809 | 0,9712 | 0,9635 |
| OvR (liblinear) | 0,9780 | 0,9779 | 0,9777 | 15,72 | 0,9823 | 0,9809 | 0,9712 | 0,9635 |
| OvR (saga) | 0,9780 | 0,9779 | 0,9777 | 10,07 | 0,9823 | 0,9809 | 0,9712 | 0,9635 |
| OvO (liblinear) | 0,9770 | 0,9770 | 0,9768 | 3,89 | 0,9758 | 0,9810 | 0,9744 | 0,9738 |

Recomendação prática:

| Cenário | Configuração | Acurácia | Tempo |
|---|---|---|---|
| Máquina acurácia | `multinomial`, `lbfgs`, `C=10` | **0,9820** | ~60s |
| Melhor custo-benefício | `ovr`, `saga`, `C=100` | **0,9790** | ~42s |
| Mínimo tempo | `OneVsOneClassifier(LR(solver='liblinear', C=100))` | **0,9780** | ~7s |

Diferença máxima entre estratégias otimizadas: apenas 0,4 pp (0,9780–0,9820).

### 5.7. Feature Engineering NLP — pontos-chave

Do estudo de feature engineering (notebook: `feature-engineering-nlp.ipynb`):

| Observação | Valor |
|---|---|
| **Hashing trick supera TF-IDF em NLP** | 0,9860 vs. 0,9770 (maior dimensionalidade ~262k e sem custo de IDF) |
| **Combinar word + char n-grams dá ganho real** | +0,5 pp (informação morfológica complementar) |
| Trees só se beneficiam de features de domain knowledge | Geo features, +1,5 pp (transforms matemáticos redundantes) |
| Regra de estilo | Pior: `hashing trick 0.9860` — ver §5.5 para contexto de cada dataset |

### 5.8. Exp1 AGNews: Classificação de Tópicos (low data)

Notebooks: `ag-news-classification.ipynb`. Teste com amostragem fixa 1000 treino / 200 test (seed 42), TF-DF 70k, lituag 2.

Resultados reais (02/07/2026, RTX 4070 ile + Intel i7):

| Modelo | Acurácia | F1 (weighted) | Precision | Recall | Tempo (s) |
|---|---|---|---|---|---|
| **DistilBERT** | **0.8350** | **0.8356** | **0.8533** | **0.8350** | 75.4 (GPU) |
| TF-IDF + LinearSVC | 0.7650 | 0.7594 | 0.7633 | 0.7650 | 0.1 (CPU) |
| TF-IDF + ExtraTrees | 0.7250 | 0.7209 | 0.7451 | 0.7250 | 0.5 (CPU) |

Early Stop (partience=2) interrompeu o treino na época 3. **DistilBERT venceu em low-data, refutando a hipótese clássica** (0.8355 vs. 0.765).

Grid Search Fino (max_features 500 – 5.000):

| max_features | LinearSVC (Acc) | ExtraTrees (Acc) |
|---|---|---|
| 500 | 0.650 | 0.690 |
| 1.000 | 0.730 | 0.730 |
| 2.000 | 0.750 | **0.745** |
| 3.000 | 0.765 | 0.730 |
| **4.000** | **0.770** | 0.725 |
| 5.000 | 0.765 | 0.740 |

![Grid Search Fino](../artifacts/grid_search_fine.png)

O ponto ótimo (1000 amostras) situa-se em **3.000–4.000 features**; valores < 1.000 perdem ~10pp (vocabulário insuficiente); valores > 4.000 adicionam ruído. LinearSVC é mais robusto a ruído (regularização L2); ExtraTrees degrada após 2.000 features (0.745→0.725) no AG News — comportamento oposto ao do Senti-Pred.

Por classe (DistilBERT): Sports F1 0.97 (fácil); World 0.85 (precisão 93% / recall 78%); Business 0.76 (recall 69%); Sci/Tech 0.75 (precision 65%, superprediz).

Análise comparativa sentir vs tópicos:

| Fator | Senti-1 (F1 ~0.98) | AG News (F1 ~0.74) |
|---|---|---|
| Tarefa | Sentimento (polaridade, vocabulário discriminativo) | Tópicos (vocabulário compartilhado: report, says, million) |
| Cardinalidade | 2–3 pólos semáticos | 4 domínios com sobreposição vocabular |
| Eficácia bigrama | 5–15% dos docs | < 1% dos docs |
| Overfit das árvores | Arvores robustas (bad → negativo) | Splits espúrios (freq. de "the" → classe errada) |
| Regularização | BO estocástica nativa | Preuseres mec global, splits binários |

ExtraTrees/RFord brilham com sinais esparsos e independentes (sentimento, dados tabulares). Em News o LinearSVC explora diferenças de frequência com pesos contínuos (0.765–0.77). A escala: com 120k amostras, DistilBERT tende a ~0.94; TF-IDF+LinearSVC satura ~0.88–0.91.

### 5.9. Multi-Task Learning (MMoE) — Google `go_emotions`

Notebook: `nlp-multi-task-classification.ipynb`. Hipótese: tarefas correlatas (Alegria, Tristeza, Raiva) se ajudam mutuamente.

- **Escassez/features fracas (TF-IDF reduzido):** compartilhar experts via MMoE eleva a performance (mitiga Transferência Negativa).
- **Interferência catastrófica com DistilBERT (Todos 43.000 stems):** redes Single-Task tornam-se autossuficientes e o MMoE se torna gargalo — **perde -0.99%** para redes isoladas.
- **Rollback tático para TF-IDF (5.000 features):** `features esparsas` como gatilhos; com F1 `macro`→`weighted`, MMoE quebrou a barreira do **0.8** → **0.9393** (+1.86% sobre Single-Task).
- **`max_features` 5k→15k:** F1-weighted **0.9464**; ganho de arquitetura cai de +1.86% → +1.24% (features mais descritivas deixam as redes isoladas mais autossuficientes).
- **`max_features` 20k:** ganho irrisório (+0.13% → 0.9477), Single-Track caiu (5k palavras extras = ruído). **Adotado 15.000 como "sweet spot".**
- **Bigramas + retenção de stopwords + limpeza URLs/ menções (15k, bigr.):** MMoE → **0.9548** (+;) vs. Single-Task 0.9461 → +0.92%.
- **Focal Loss binária:** F1-weighted MMoE → **0.9566** (T, occasionally 0.962+).
- **Duelo final com Deep Learning clássico (features esparsas):** LightGBM 0.9473 (sofre com alta dimensionalidade), **LinearSVC 0.9572**, **vancedor ExtraTrees 0.9643 F1-weighted — árvores randomizadas vançam matriz sparsity esparsa** em alta dimensionalidade, sem GPU.

## 6. Discussão

**A "relatividade" dos modelos (No Free Lunch):** não existe um modelo universal. O LinearSVC variou de **0.74** (F1-Macro) a **0.94** apenas por ajustes de vocabulário e n-grams; o KNN superou frameworks complexos de AutoML em um caso; o Ensemble Pyramid superou os individuais com 0.98+.

**O poder da engenharia de features:** bigramas capturam a negação ("não é bom"), e vocabulário de equilíbrio (sweet spot do dataset) importa: 15k features no Senti- (mix large, corpus), 3–4k no AG News (1000 amostras). A regra empírica: **Σ documentsa ~ Σ termos candidatos ~ max_features ideal**.

**Deep Learning vs. Clássicos:** em regimes de dados abundantes e features TF-IDF de alta dimensionalidade, modelos lineares/árêtes (Spark) superam transformers e redes profundas; em baixas amostragens, o tunig regularizado do transformer vence. Dados completos é obrigatório para redes neurais (Tensor network ganham +16–17pp de 30k→74k).

**Preprocessamento e a "Data-Centric AI":** o duelo A vs B mostra que o tratamento de hashtags, pontuação e números é mais decisivo que o modelo — engenharia da limpeza ganhou **+0.40%**.

**Limitações/biases:** o Mamba ficou **TBD** (dependência de hardware CUDA estrita); Sentence-BERT frozen é inadequado para polaridade (limite de representação, dados não resolvem); a exatidão dos valores depende da seed (42) e rfira limitação de contexto e hardware; o Mamba,O dataset go_emotions tem dominância da classe "Alegria".

## 7. Conclusões e Recomendações

- **Baseline rápida:** TF-IDF (70k, bigrama, sublinear) + LinearSVC — 0.98 classificação de sentimento em 4; para Ao e-tralização pointer.
- **Quando o custo computacional importa:** LinearSVC/ExtraTrees sobre TF-IDF esparso — sem GPU, segundos de treino.
- **Quando a precisão é requisito (>0.98):** fine-tune DistilBERT no dataset completo (40 min de GPU, às 0.9710) ou Ensemble Pyramid (~0.98+).
- **Sem GPU / orçamento moderado:** TextCNN (0.9530 em 13s).
- **Low-data (N≤1000):** fine-tune regularizado (early stopping activado) supera clássico — testar ambas abordagens.
- **Multi-tarefa:** preferir TF-IDF 15k + bigramas + Focal Loss com MMoE quando features são fracas; evitar MMoE com embeddings densos "fartos" (interferência catastrófica).
- **Engenharia de dados > modelo:** priorizar a limpeza (hashtags/pontuação/contrarições) e n-grams antes de trocar de arquitetura.

## 8. Referências e Arquivos

- `ag-news-classification.ipynb` — Exp1 AG News (low data, grid search).
- `twitter-sentiment-analysis.ipynb` — Pipeline B.
- `senti-pred_pipeline.ipynb` — Pipeline A.
- `pipelines_abc_comparison/` — comparativo A vs B vs C (remake2) com what-ifs (n-grams, vocabulário, pré-processamento, modelo; McNemar).
- `logistic-regression-multiclass.ipynb` — estratégias multiclassific Logistics Regression.
- `feature-engineering-nlp.ipynb` — feature engineering alguma NLP.
- `nlp-multi-task-classification.ipynb` — MMoE multi-finition (go_emotions).
- `../NLP-twitter-methods-comparasion.ipynb` — Twitter Methods Comparison (5 paradigmas).
- `../ensemble_pyramid.py` — Ensemble Pyramid / Versatile Ensemble Pyramid (AutoML RL).
- Referências: Devlin et al. (BERT); Sanh et al. (DistilBERT); Gu & Dao et al. (Mamba — SSMs); Sennrich? ver papers de MMoE (Ma et al., SIGIR 2018) e Lin et al. (Focal Loss, ICCV 2017).