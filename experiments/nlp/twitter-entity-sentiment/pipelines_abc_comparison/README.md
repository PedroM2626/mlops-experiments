# Senti-Pred — Comparação Rigorosa dos Pipelines A, B e C (Senti-Pred-remake2)

> **Área:** NLP
> **Tarefa:** Classificação de sentimentos em 4 classes (Irrelevant, Negative, Neutral, Positive)
> **Métrica principal:** F1-Macro (e Acurácia / F1-Weighted como complementares)
> **Status:** Concluído
> **Datasets:** Twitter Entity Sentiment — `twitter_training.csv` (73.996 linhas úteis) / `twitter_validation.csv` (1.000 linhas)
> **Data de execução:** 12/08/2026 — seed 42 — CPU (16 cores lógicos, Windows, sklearn 1.7.1)

---

## 1. Resumo

Este estudo expande o duelo de engenharia "Pipeline A vs. Pipeline B" (documentado no
`experiments/nlp/README.md`, §5.4) com uma **terceira pipeline** — a **Pipeline C
(Senti-Pred-remake2)**, recordista histórica de 97,80% de acurácia/F1. As três pipelines
foram reimplementadas fielmente a partir dos seus artefatos de origem e executadas sob o
**mesmo dataset, mesmo split e mesma seed**, garantindo comparação justa. Além da
reprodução canônica (F1-Macro: **A = 0,9845** com ExtraTrees; **B = 0,9833** com LinearSVC C=19;
**C = 0,9773** com Voting LinearSVC+LR), o trabalho sistematiza um extenso conjunto de
**"what-ifs"** (ablações controladas) sobre o número de n-gramas, o tamanho do vocabulário,
`min_df`, `sublinear_tf`, o pré-processamento e o modelo. A conclusão central reforça o
paradigma *Data-Centric AI*: **a limpeza de texto importa mais do que a escolha do modelo**,
e o vetorizador da Pipeline C (100k features, 4-gramas) é o melhor componente individual,
mas a limpeza da Pipeline C é a menos favorável das três — o melhor F1-Macro observado
(**0,9857**) surge quando o pré-processamento **A** é combinado com o vetorizador **C**.

## 2. Contexto e Objetivos

O repositório contém um grupo de experimentos NLP que compara, entre outras coisas, duas
variações de pipeline de análise de sentimento em redes sociais (Twitter):

- **Pipeline A** (`senti-pred_pipeline.ipynb`) — pré-processamento **agressivo**
  (remove URLs, menções, hashtags inteiras, pontuação e dígitos).
- **Pipeline B** (`twitter-sentiment-analysis.ipynb`) — pré-processamento **conservador**
  (conserva conteúdo de hashtags, pontuação `!?.,'"`, hífens e números).

A seção §5.4 do README de NLP concluiu que **B = 0,9860 F1-Weighted vs A = 0,9820**
com LinearSVC C=19 — um ganho de **+0,40 pp** atribuído exclusivamente à limpeza.

Em paralelo, o projeto `experiments/senti-pred-variations/` documenta a evolução de um
pipeline "remake2" (`Senti-Pred-remake2`) que atingiu o recorde de **97,80%** (acurácia/F1)
com **TF-IDF (100k) + 4-gramas + Voting (LinearSVC + LogisticRegression)**.

**Objetivo deste experimento:** unificar as duas linhas de pesquisa, trazendo o
Senti-Pred-remake2 como **Pipeline C** para dentro do comparativo A vs B, e responder de
forma rigorosa às perguntas de "what-if":

1. *E se Eu aumentar/diminuir o número de bigramas (n-gramas) da Pipeline C, A ou B?*
2. *E se Eu mudar o pré-processamento de cada pipeline (hashtags, pontuação, dígitos,
   stopwords, lematização, contrações)?*
3. *E se Eu aumentar/diminuir o tamanho do vocabulário (max_features), min_df ou
   sublinear_tf?*
4. *E se Eu trocar o modelo (voting hard/soft, class_weight, LinearSVC isolado)?*
5. *As diferenças entre as pipelines são estatisticamente significativas?*

**Hipóteses testadas:**

- `H1` — Bigramas são cruciais; remover o bigramas derruba o F1-Macro em todas as pipelines.
- `H2` — A Pipeline C tem o melhor vetorizador, mas a limpeza mais "pesada";
  combinações cruzadas (pré-processamento X + vetorizador Y) podem superar as canônicas.
- `H3` — Diferenças inferior a ~1 pp entre pipelines no holdout (N=1.000) **não** são
  estatisticamente significativas (teste de McNemar).

## 3. Fundamentação Teórica (curta)

- **TF-IDF** — matriz esparsa onde cada dimensão é um termo (ou n-grama); peso = `tf × idf`
  com amortecimento logarítmico quando `sublinear_tf=True` (`1 + log(tf)`).
- **n-gramas de palavras** — unigramas capturam o léxico; **bigramas** capturam negação e
  composição (`not good`, `very bad`) — exatamente o que é decisivo em sentimento. N-gramas
  de ordem maior (3–5) adicionam vocabulário raro e ruído.
- **LinearSVC** — SVM linear com penalidade L2 (`C`); robusto em espaço esparso
  alta-dimensional; não tem `predict_proba` (logo, para `voting='soft'`, precisou de
  calibração Platt via `CalibratedClassifierCV`).
- **Voting Ensemble** — combinação democrática; `hard` = voto por classe majoritária,
  `soft` = média de probabilidades.
- **class_weight='balanced'** — re-pondera classes desbalanceadas pelo inverso da frequência.
- **Pré-processamento Data-Centric** — limpeza de ruído (URLs, menções), decisão sobre
  hashtags (remover/bloco vs. preservar o conteúdo), pontuação com carga de sentimento
  (`!`, `?`), stopwords (em inglês; `not`/`no` preservados por serem sinal), lematização
  WordNet e expansão de contrações.
- **Teste de McNemar** — teste pareado (exato, binomial) sobre a validação, para decidir se
  duas classificações diferem significativamente.

## 4. Metodologia

### 4.1 Dados

| Split | Fonte | Linhas (após limpeza de carga) | Classes |
|---|---|---|---|
| Treino | `twitter_training.csv` | 73.996 | 4 |
| Validação | `twitter_validation.csv` | 1.000 | 4 |

- Remoção de linhas com `text` ou `sentiment` nulos e de textos que ficam vazios após a limpeza.
- Classes no treino: Negative 22.542 / Positive 20.832 / Neutral 18.318 / Irrelevant 12.990.

### 4.2 Pré-processamento das três pipelines

| Componente | Pipeline A (agressiva) | Pipeline B (conservadora) | Pipeline C (remake2) |
|---|---|---|---|
| URLs / www | removidas | removidas | removidas |
| Menções `@user` | removidas | removidas | removidas |
| Hashtags | removidas (`#palavra` inteira) | conteúdo preservado (`#great`→`great`) | símbolo `#` removido, palavra preservada |
| Pontuação | toda removida | preservada `!?.,'"` e `-` | preservados apenas `!` e `?` |
| Números | removidos | preservados | removidos |
| Stopwords | não removidas (fase final) | não removidas | removidas, **preservando `not`/`no`** |
| Lematização | não usada (fase final) | não usada | WordNet |
| Contrações | não expandidas | não expandidas | expandidas (`n't`→` not`, etc.) |
| Caixa | minúscula | minúscula | minúscula |

### 4.3 Vetorizadores canônicos

| Pipeline | max_features | ngram_range | min_df | sublinear_tf | token_pattern |
|---|---|---|---|---|---|
| A | 70.000 | (1,2) | 2 | True | default |
| B | 70.000 | (1,2) | 2 | True | default |
| C | 100.000 | (1,4) | 2 | True | `\w{1,}` |

### 4.4 Modelos canônicos (por pipeline)

- **A:** LR(2000 it.), MultinomialNB, LinearSVC (C=1; 10; 19), ExtraTrees(100). Campeã documentada: **ExtraTrees**.
- **B:** LR(C=11), ExtraTrees, LinearSVC(C=19), PassiveAggressive, KNN(7, cosine), Ridge, SGD(modified_huber). Campeã: **LinearSVC C=19**.
- **C:** LinearSVC(C=0.5, `class_weight=balanced`), LR(C=10, balanced), **Voting hard (SVC+LR)**, conforme artefato oficial.

### 4.5 Desenho experimental ("what-ifs")

Sempre que uma dimensão é variada, as demais permanecem canônicas. Modelo "champion" por
pipeline nas ablações: **A/B → LinearSVC C=19**; **C → Voting** (para alinhar com a
documentação e isolar o efeito de features).

| Exp | Dimensão | Valores testados |
|---|---|---|
| E1 | Reprodução canônica | todos os modelos de cada pipeline |
| E2 | **n-gramas** | (1,1), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3) |
| E3 | `max_features` | 10k, 25k, 50k, 70k, 100k, 150k, 200k |
| E4 | `min_df` | 1, 2, 3, 5 |
| E5 | `sublinear_tf` | True, False |
| E6 | Toggles de pré-processamento | ver §5.4 |
| E7 | Modelo da Pipeline C | SVC isolado, LR isolado, Voting hard/soft, com/sem `class_weight`, C=0.5 vs 19 |
| E8 | `class_weight=balanced` justiça | aplicado a A/B/C (LR C=10 e SVC C=19) |
| E9 | Cross pré-processamento × vetorizador | 3 cleaners × ({A,B,C} vetorizadores) |

### 4.6 Avaliação (protocolo)

- **Holdout fixo** treino/validação original do dataset (73.996 / 1.000), **seed 42**.
- Métricas: **Acurácia, F1-Macro, F1-Weighted**, F1 por classe, matriz de confusão, tempo de
  treino, `n_features`.
- Teste de significância pareado **McNemar exato** (binomial) entre as três canônicas.
- Hardware: Windows, 16 núcleos lógicos, Python 3.13.1, scikit-learn 1.7.1.

### 4.7 Reprodução

```bash
cd experiments/nlp/pipelines_abc_comparison
python run_abc_comparison.py --out ../../artifacts/pipelines_abc_<timestamp>_<sha>
```

- Saída: `experiments/artifacts/pipelines_abc_20260812_123257_0295952/`
  (`results_*.csv`, `results_all.csv`, `fig*.png`, `predictions.npz`,
  `val_with_predictions.csv`, `champions_summary.csv`, `meta.json`).
- Código: `pipelines_abc_core.py` (cleaners/vectorizadores/modelos/avaliação),
  `run_abc_comparison.py` (orquestração das ablações).

## 5. Resultados

> Todos os valores abaixo foram **medidos nesta execução** (não copiados das documentações).
> Diferenças de ±0,1 pp em relação aos READMEs originais refletem variação de versão dos
> dados/seed; a hierarquia qualitativa reproduz-se.

### 5.1 E1 — Reprodução canônica (F1-Macro por modelo)

| Pipeline | Modelo | Acc | F1-Macro | F1-Weighted | Tempo (s) |
|---|---|---|---|---|---|
| **A** | **ExtraTrees** | **0,985** | **0,9846** | **0,9850** | 121,2 |
| A | LinearSVC C=10 | 0,983 | 0,9828 | 0,9830 | 11,6 |
| A | LinearSVC C=19 | 0,981 | 0,9807 | 0,9810 | 15,4 |
| A | LinearSVC C=1 | 0,980 | 0,9797 | 0,9800 | 6,8 |
| A | LR | 0,975 | 0,9743 | 0,9750 | 10,8 |
| A | MultinomialNB | 0,914 | 0,9105 | 0,9134 | 2,4 |
| **B** | **LinearSVC C=19** | **0,983** | **0,9833** | **0,9830** | 17,3 |
| B | Ridge | 0,983 | 0,9830 | 0,9830 | 4,5 |
| B | LR C=11 | 0,981 | 0,9813 | 0,9810 | 13,6 |
| B | PassiveAggressive | 0,980 | 0,9806 | 0,9801 | 3,4 |
| B | KNN cosine | 0,978 | 0,9785 | 0,9781 | 4,3 |
| B | ExtraTrees | 0,977 | 0,9772 | 0,9770 | 119,9 |
| B | SGD | 0,977 | 0,9769 | 0,9770 | 3,0 |
| **C** | **LinearSVC C=0.5** | **0,978** | **0,9782** | **0,9780** | 6,4 |
| C | Voting hard (SVC+LR) | 0,978 | 0,9773 | 0,9780 | 21,9 |
| C | LR C=10 | 0,978 | 0,9773 | 0,9780 | 19,6 |

*Entre as campeãs de cada pipeline:* **A (ExtraTrees) 0,9846 > B (LinearSVC C=19) 0,9833
> C (Voting) 0,9773.**

### 5.2 E2 — What-if do número de n-gramas (max N) — **pergunta central do estudo**

Δ (em pp = pontos percentuais) frente à configuração canônica de cada pipeline
(→ *(1,2)* para A/B; *(1,4)* para C):

| N-gramas | Δ A | Δ B | Δ C |
|---|---|---|---|
| (1,1) — **só unigramas** | **−2,57** | **−3,09** | **−4,76** |
| (1,2) | 0,00 (canônico) | 0,00 (canônico) | **+0,33** |
| (1,3) | +0,00 | −0,46 | +0,00 |
| (1,4) | −0,11 | −0,49 | 0,00 (canônico) |
| (1,5) | −0,11 | −0,61 | −0,46 |
| (2,2) — só bigramas | −2,48 | −2,81 | −1,32 |
| (2,3) | −3,73 | −3,80 | −3,48 |

**Leituras:**

- **Bigramas são imprescindíveis (H1 confirmada):** remover os bigramas derruba **−2,6 a −4,8 pp**.
- **Pipeline C está "super-ajustada" a 4-gramas:** reduzindo de (1,4) para **(1,2) o
  F1-Macro SOBE +0,33 pp** (0,9773 → 0,9806). Os 4-gramas da remake2 agregam ruído.
- **Pipeline B é a mais sensível a n-gramas** de ordem superior: qualquer N>2 a degrada (−0,5 a −0,6 pp).
- **Pipeline A é a mais robusta** ao aumento de ordem (estável até trigramas).

### 5.3 E3–E5 — What-ifs de vocabulário e normalização TF

**`max_features` (Δ em pp vs canônica):**

| max_features | Δ A | Δ B | Δ C |
|---|---|---|---|
| 10.000 | −4,02 | −4,63 | **−8,47** |
| 25.000 | −0,66 | −1,70 | −2,55 |
| 50.000 | −0,23 | −0,38 | −0,91 |
| 70.000 | 0,00 | 0,00 | −0,44 |
| 100.000 | +0,39 | +0,18 | 0,00 (canônico) |
| 150.000 | +0,39 | +0,15 | +0,09 |
| 200.000 | +0,30 | +0,24 | **+0,41** |

- Vocabulário é o recurso mais "elástico" da Pipeline C: cortar para 10k custa **−8,5 pp**;
  expandir para 200k rende **+0,41 pp** (0,9773 → 0,9815). A e B também ganham levemente com
  vocabulário > canônico (100–200k).

**`min_df`:** A é insensível (0,00 em todos); **B melhora com `min_df=1` (+0,30 pp)**;
**C piora com `min_df=5` (−0,67 pp)**.

**`sublinear_tf`:** neutro em A e C; **B ligeiramente melhor sem sublinear (+0,12 pp)**.

### 5.4 E6 — What-if de pré-processamento (toggles 1-a-1)

**Pipeline A**

| Toggle | F1-Macro | Δ (pp) |
|---|---|---|
| A default (tudo removido) | 0,9807 | 0,00 |
| **Manter hashtags** | 0,9848 | **+0,42** |
| **Manter pontuação** | 0,9848 | **+0,41** |
| **Manter dígitos** | 0,9848 | **+0,42** |
| Manter pontuação+dígitos | 0,9841 | +0,35 |

→ **Limpeza "agressiva" é agressiva demais nesses eixos:** preservar o conteúdo de hashtags,
pontuação ou números rende **≈ +0,4 pp** cada à Pipeline A.

**Pipeline B**

| Toggle | F1-Macro | Δ (pp) |
|---|---|---|
| B default | 0,9833 | 0,00 |
| Remover pontuação | 0,9851 | **+0,18** |
| Remover hashtags | 0,9812 | −0,20 |
| Remover dígitos | 0,9821 | −0,12 |

→ B já é bem calibrada; apenas *remover pontuação* dá leve ganho (+0,18 pp); perder
hashtags/dígitos custa.

**Pipeline C**

| Toggle | F1-Macro | Δ (pp) |
|---|---|---|
| C default | 0,9773 | 0,00 |
| **Manter stopwords** | 0,9803 | **+0,30** |
| Não expandir contrações | 0,9730 | **−0,43** |
| Remover palavra do hashtag | 0,9741 | −0,32 |
| Remover `!` `?` | 0,9762 | −0,12 |
| Não lematizar | 0,9769 | −0,05 |

→ **A limpeza da Pipeline C é a menos favorável no comparativo:** manter as stopwords
recupera +0,30 pp (as stopwords removidas eram, no fundo, sinal), expandir contrações é
essencial (+0,43 pp se perdido), e o conteúdo dos hashtags carrega sentimento (+0,32 pp).

### 5.5 E7 — What-if de modelo na Pipeline C

| Modelo | F1-Macro | Acc |
|---|---|---|
| **LinearSVC C=0.5 (isolado)** | **0,9782** | 0,978 |
| Voting hard sem weight | 0,9779 | 0,978 |
| LinearSVC C=0.5 sem weight | 0,9779 | 0,978 |
| **Voting hard (oficial)** | 0,9773 | 0,978 |
| LR C=10 balanced | 0,9773 | 0,978 |
| Voting soft (calibrado) | 0,9758 | 0,976 |
| LR C=10 | 0,9746 | 0,975 |
| LinearSVC C=19 | 0,9741 | 0,975 |

→ **o Voting oficial não é superior ao LinearSVC C=0.5 isolado** (−0,09 pp); votação `soft`
com SVC calibrado piora; `class_weight=balanced` praticamente neutro; **`C=0.5` ≫ `C=19`**
no regime com `balanced` (0,9782 vs 0,9741).

### 5.6 E8 — Fairness `class_weight=balanced`

| Config | F1-Macro A | F1-Macro B | F1-Macro C |
|---|---|---|---|
| LR C=10 balanced | 0,9788 (−0,15) | 0,9809 (−0,23) | 0,9773 (0,00) |
| SVC C=19 balanced | 0,9795 (−0,12) | 0,9815 (−0,18) | 0,9733 (−0,04) |

→ aplicar `balanced` **não é** a fonte da performance da C: para A/B a ponderação até
degrada ~0,1–0,2 pp; a C, com C=19, também perde um pouco. A vantagem da C vem do
**C=0.5 + balanced combinados**, não do peso sozinho.

### 5.7 E9 — Cross pré-processamento × vetorizador (modelo champion do pré-processamento)

| Pré-processamento | Vetorizador | F1-Macro | Acc |
|---|---|---|---|
| **A** | **C (100k, 1-4)** | **0,9857** | 0,986 |
| **B** | C (100k, 1-4) | 0,9833 | 0,984 |
| A | A/B (70k, 1-2) | 0,9807 | 0,981 |
| B | A/B (70k, 1-2) | 0,9833 | 0,983 |
| C | C (100k, 1-4) | 0,9773 | 0,978 |
| C | A/B (70k, 1-2) | 0,9771 | 0,978 |

→ **o melhor single-run do estudo é `pré-processamento A + vetorizador C` = 0,9857
F1-Macro**, superando TODAS as canônicas. O vetorizador da remake2 é o mais poderoso;
a limpeza A é a mais compatível com ele.

### 5.8 Significância estatística (McNemar exato) e F1 por classe

F1 por classe (campeãs):

| Classe | A | B | C |
|---|---|---|---|
| Positive | 0,9856 | 0,9767 | 0,9693 |
| Negative | 0,9887 | 0,9868 | 0,9831 |
| Neutral | 0,9843 | 0,9842 | 0,9859 |
| Irrelevant | 0,9796 | 0,9854 | 0,9711 |
| **Erros /1000** | **15** | **17** | **22** |

McNemar (pareado, exato):

| Par | Discordantes (p1 ok / p2 ok) | p-valor | Conclusão |
|---|---|---|---|
| A vs B | 10 / 9 | 1,00 | n.s. |
| A vs C | 11 / 6 | 0,33 | n.s. |
| B vs C | 11 / 10 | 1,00 | n.s. |

→ **H3 confirmada:** com N=1.000 na validação, diferenças de ~0,6–0,7 pp entre pipelines
são **estatisticamente indistinguíveis** (p ≥ 0,33). A ordenação vista reflete tendência,
não diferença provada.

### 5.9 Diagnóstico de overfitting e *dataset shift*

Para além da validação, foram medidas: (i) F1 no próprio treino após ajuste completo e
(ii) CV-5 estratificado sobre o treino. O "gap de overfitting" é definido como
`F1_treino − F1_validação` (positivo ⇒ overfit; ~0 ⇒ generalização perfeita); o "gap de
*dataset shift*" como `F1_validação − F1_mediano_CV`.

| Campeã | TREINO F1-Macro | VALIDAÇÃO F1-Macro | CV-5 F1-Macro | Overfit gap (pp) | Shift gap (pp) |
|---|---|---|---|---|---|
| A: ExtraTrees (100) | 0,9781 | 0,9845 | 0,9255 ± 0,0022 | **−0,64** (n.s.) | **+5,91** |
| B: LinearSVC C=19 | 0,9763 | 0,9833 | 0,9186 ± 0,0015 | **−0,70** (n.s.) | **+6,47** |
| C: Voting (oficial) | 0,9697 | 0,9773 | 0,9200 ± 0,0019 | **−0,76** (n.s.) | **+5,74** |
| C: LinearSVC C=0.5 (single) | 0,9660 | 0,9782 | 0,9222 ± 0,0019 | **−1,23** (n.s.) | **+5,61** |

**Leituras:**

1. **Nenhuma pipeline sofre de overfitting.** O *overfit gap* é **negativo** em todos os casos
   (entre −0,64 e −1,23 pp), i.e., `F1_treino < F1_validação`. O modelo erra *mais* no treino
   do que na validação — sinal clássico de **underfitting leve** (o treino é mais difícil e
   contém exemplos ruidosos/ambíguos), nunca de memorização.
2. **Há um forte *dataset shift* entre treino e validação (~+6 pp).** A validação é
   substancialmente **mais fácil** do que uma média de folds dentro do treino
   (`F1_CV ≈ 0,92` vs `F1_val ≈ 0,98`). Consequência: **os ~0,98 reproduzidos para todas as
   pipelines são inflados**; a capacidade de generalização honesta (CV) situa-se em ~0,92.
   A ranking por valor absoluto na validação reflete *facilidade do conjunto de teste*, não
   qualidade intrínseca do modelo.
3. **Ranking por generalização honesta (CV-5):** A (ExtraTrees) **0,9255** > C (LinearSVC
   C=0.5) 0,9222 ≈ C (Voting) 0,9200 > B (LinearSVC C=19) 0,9186. A ordem **A > C > B** é
   diferente da ranking por validação pura (`A > B > C`), e reforça a recomendação de usar a
   Pipeline A quando possível — sob validação cruzada, ExtraTrees é a melhor generalizadora.
4. **Variance do CV é baixíssima** (~0,002), logo a diferença A vs B/C (~0,7 pp) no CV já é
   mais robusta do que a comparação pareada na validação. Em produção, **ExtraTrees sobre A
   (ou sobre o cross A+vetorizador C) é a escolha mais defensável em termos de
   generalização**, não LinearSVC C=19 sobre B.
5. **Recomendação metodológica:** reportar **CV estratificado sobre o treino** (e não apenas
   o holdout original) como métrica primária; armazenar `overfit_gap` e `shift_gap` nos
   metadados de cada modelo no MLflow.

→ Verificação adicional de *sanity*: o classificador majoritário (Negative) acerta 30,2 %
no treino e 26,6 % na validação, confirmando que as classes estão apenas *moderadamente*
desequilibradas e que a base aleatória é ~25 % — os ~98 % observados não são artefato de
desbalanceamento extremo.

### 5.10 Diagnóstico Qualitativo de Generalização (Frases Reais Fora de Domínio)

Para aferir a real compreensão semântica dos modelos (Pipeline B e C com LinearSVC) em situações fora da distribuição (Out-of-Distribution) do dataset de Twitter, foram testadas 5 frases contendo sarcasmo, negação, sentenças neutras e contextos não relacionados a marcas. Os resultados ilustram os limites práticos do TF-IDF:

* **Dupla Negação ("The new update is not bad at all, I actually think it is quite good.")**:
  **Sucesso**. Ambos B e C preveram *Positive*. Bigramas provam seu valor capturando "not bad".
* **Sarcasmo ("I absolutely love waiting 3 hours in line for a coffee... best day ever.")**:
  **Falha Grave**. Ambos preveram *Positive*. O TF-IDF apenas soma os pesos elevados das palavras "love" e "best day", falhando em capturar a estrutura irônica.
* **Neutro Absoluto ("I have no strong feelings about this movie, it was just okay.")**:
  **Falha**. Pipeline B previu *Negative* (peso na palavra "no"), Pipeline C previu *Positive* (peso na palavra "okay"). Modelos estatísticos perdem o centro se as palavras puxam para os polos.
* **Trivial OOD ("What is the weather going to be like tomorrow?")**:
  **Falha (Alucinação)**. Ambos preveram *Positive*. Sendo o dataset enviesado em entidades e marcas, linguagem puramente casual e investigativa sofre predições arbitrárias (alucinação estatística).
* **Negativo Complexo ("My flight got delayed and my luggage is lost. I am furious!")**:
  **Misto**. Pipeline B previu *Negative* (sucesso, preservou a pontuação '!' forte). Pipeline C previu *Neutral* (falha, limpou demais a sentença e não conectou o jargão de voo como negativo absoluto).

**Conclusão qualitativa:** Embora alcancem ~98% no domínio de treino/validação, no mundo aberto os classificadores de TF-IDF são limitados. Eles não possuem entendimento profundo de semântica e sofrem com sarcasmo e sentenças OOD, funcionando mais como "balanças de palavras-chave" hiper-otimizadas do que verdadeiros interpretadores de linguagem natural (como LLMs modernos).

## 6. Discussão

1. **O vetorizador da Pipeline C (remake2) é o ativo mais valioso.** Seus 100k features com
   4-gramas, quando casados com o pré-processamento A, geram o melhor resultado de todo o
   estudo (**0,9857**), +0,50 pp sobre a canônica A. O ganho do *remake2* veio muito mais do
   "vocabulário extremo" do que da limpeza.
2. **A limpeza da Pipeline C é sua maior fragilidade.** A remoção de stopwords custa
   **+0,30 pp** (se mantidas), os 4-gramas custam **+0,33 pp** (se reduzidos a bigramas), e
   juntas essas duas mudanças — *manter stopwords + bigramas* — recolocariam a C na faixa
   dos A/B em validade externa.
3. **Há um trade-off claro entre "poder de features" e "ruído".** Pipelines com vocabulário
   pequeno (A/B 70k) são mais sensíveis a `min_df`/`sublinear`; a C, com 100–200k, depende
   criticamente de `max_features` (−8,5 pp se cortado a 10k).
4. **Modelo é o fator menor** (*No Free Lunch* semântico): a mesma pipeline muda ≤0,5 pp
   trocando de modelo, enquanto a mesma família de modelo muda até +0,4 pp trocando de limpeza;
   o `Voting` oficial da C é neutro/levemente pior que `LinearSVC C=0.5` isolado.
5. **Limitação importante — poder estatístico.** O holdout tem apenas 1.000 amostras;
   com ~15–22 erros totais, o McNemar não consegue separar pipelines que diferem < 1 pp.
   Para decidir "qual é melhor" em produção, seria necessária **validação cruzada repetida**
   ou um teste maior.
6. **Nenhum overfitting; forte *dataset shift*.** O diagnóstico completo (§5.9) mostra que
   nenhuma campeã memoriza (`F1_treino < F1_validação` em todas, gap −0,6 a −1,2 pp). Porém
   a validação é **~6 pp mais fácil** do que a CV-5 sobre o treino (`F1_CV ≈ 0,92`), o que
   inflaciona todos os "0,98" reportados. A ranking honesta (CV) é **A (ExtraTrees 0,9255) >
   C (Voting/LinearSVC 0,9200–0,9222) > B (LinearSVC C=19 0,9186)** — em produção, prefira
   a Pipeline A (ou o cross A+vetorizador C).
7. **Consistência com a documentação existente.** A hierarquia *Data-Centric* (limpeza >
   modelo) e o recorde do remake2 (~97,8% na validação) se reproduzem; mas a interpretação
   atribuída — "4-gramas + 100k explicam o recorde" — deve ser qualificada: o componente que
   mais explicava o resultado *na validação* era o vetorizador, não a limpeza; e a validação
   em si é um alvo facilitado (~6 pp acima da CV).

## 7. Conclusões e Recomendações

- **Melhor por validação (F1-Macro na validação, 1.000 amostras):** Pipeline **A + ExtraTrees**
  = 0,9845. **Melhor *single run* (combinação de componentes):** **pré-processamento A +
  vetorizador C + LinearSVC C=19** = 0,9857. **Melhor custo/benefício:** Pipeline **B +
  LinearSVC C=19** = 0,9833 em ~15 s (ExtraTrees custa ~120 s por ~+0,1 pp).
- **Melhor por generalização honesta (CV-5 estratificado sobre o treino):** Pipeline
  **A + ExtraTrees** = 0,9255 (estável, ±0,002). Esse é o critério recomendado para produção,
  pois a validação original sofre de *dataset shift* (~+6 pp).
- **Nenhuma pipeline sofre de overfitting**: todas têm `F1_treino < F1_validação`
  (`overfit_gap` negativo de −0,6 a −1,2 pp), o que indica leve underfitting sobre um treino
  mais ruidoso — não há memorização.
- **Para máxima acurácia neste domínio:** usar **pré-processamento A + vetorizador C
  (100k, bigramas, sublinear) + LinearSVC/ExtraTrees** → F1-Macro ≥ 0,985 na validação;
  apenas **0,2–0,3 pp** abaixo do estado-da-arte prático (~0,987) com custo de segundos.
- **Não copiar a limpeza da remake2 às cegas:** remover stopwords e usar 4-gramas penalizam
  em regime de validação externa; **manter stopwords e usar bigramas** agrega ~+0,6 pp à C.
- **Priorizar bigramas primeiro, depois vocabulário.** Bigramas valem **+2,6 a +4,8 pp**
  (nunca abrir mão deles); amplitude do vocabulário vale **+0,2 a +0,4 pp** acima do ponto
  de saturação — e evitar truncamento abaixo de ~50k na C (−0,9 pp em 50k, −8,5 pp em 10k).
- **Modelo:** preferir LinearSVC (C ≈ 0,5–19) ou **ExtraTrees** sobre TF-IDF (ExtraTrees
  generaliza melhor em CV); o voting hard oficial da remake2 é dispensável e `voting='soft'`
  deve ser evitado com SVC linear.
- **Em termos de MLOps:** manter modularidade (cleaners parametrizáveis), logar os toggles
  como hiperparâmetros no MLflow, **reportar CV estratificado + overfit_gap + shift_gap** como
  métricas complementares à validação, e tratar as diferenças finais com testes pareados
  (McNemar) antes de "campear" uma pipeline em produção.
- **Limite superior prático documentado para novas explorações:** `max_features=200k`
  (+0,41 pp vs 100k em C), `min_df=1` (+0,30 pp em B), pré-processamento A + vetorizador C.

## 8. Referências e Arquivos

- `run_abc_comparison.py`, `pipelines_abc_core.py` — código do experimento (nesta pasta).
- `experiments/artifacts/pipelines_abc_20260812_123257_0295952/` — CSVs de resultados,
  figuras `fig1..fig5`, predições, matriz de confusão e summary.
- Notebooks-fontes: `experiments/nlp/senti-pred_pipeline.ipynb` (A),
  `experiments/nlp/twitter-sentiment-analysis.ipynb` (B),
  `experiments/senti-pred-variations/Senti-Pred-remake2/` (C).
- Documentação correlata: `experiments/nlp/README.md` (§5.3–5.6),
  `experiments/senti-pred-variations/README.md`, `EXPERIMENTS_SUMMARY.md`.
- Métodos: Vapnik (SVM), Manning et al. (TF-IDF/n-grams), McNemar (1947);
  calibração Platt (1999).

## Scripts e Reprodu��o

# Senti-Pred Pipelines A vs B vs C — Scripts

Código do estudo comparativo das 3 pipelines de sentimento do Twitter.

## Arquivos

| Arquivo | Função |
|---|---|
| `pipelines_abc_core.py` | Reimplementação fiel dos 3 pré-processamentos, vetorizadores, modelos e função de avaliação |
| `run_abc_comparison.py` | Orquestração das 9 baterias de experimentos (E1–E9) e exportação de CSV/JSON |
| `README.md` | Documentação acadêmica do estudo (resultados, what-ifs, discussão, conclusões) |

## Reproduzir

```bash
# do diretório experiments/nlp/pipelines_abc_comparison
python run_abc_comparison.py --out ../../artifacts/pipelines_abc_<timestamp>_<sha>
```

Para rodar apenas um subconjunto de baterias:

```bash
python run_abc_comparison.py --stages canonical ngrams --out <dir>
```

Etapas disponíveis (`--stages`): `canonical ngrams max_features min_df sublinear_tf
preprocessing model_c fairness cross`.

## Dependências

`pandas`, `numpy`, `scikit-learn>=1.0`, `joblib`, `nltk` (punkt, stopwords, wordnet, omw-1.4).
Requires the raw CSVs in
`experiments/senti-pred-variations/senti-pred-exp1/data/raw/`.

## Saída

Todos os artefatos vão para `experiments/artifacts/pipelines_abc_<timestamp>_<sha>/`:

- `results_<etapa>.csv` e `results_all.csv` — tabelas de métricas por execução.
- `fig1_canonical.png` … `fig5_model_c.png` — figuras comparativas.
- `champions_summary.csv`, `predictions.npz`, `val_with_predictions.csv` — dados das campeãs.
- `meta.json` — metadados (seed, tamanhos, hardware, data).
