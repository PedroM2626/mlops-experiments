# Experimentos Hierárquicos — 20 Newsgroups

> **Área:** NLP + Hierarquia de Classes
> **Tarefa:** Classificação hierárquica (supervisionada) e Clustering hierárquico (não supervisionado)
> **Métrica principal:** Acurácia exact-match de folha; Purity / NMI / ARI / F (clustering); HF (hierárquica)
> **Status:** Concluído
> **Datasets:** 20 Newsgroups (scikit-learn) — ~18.000 mensagens reais de Usenet, 1993, 7 parentes / 20 classes

## 1. Resumo

Dois experimentos complementares exploram estruturas em árvore (`parente → folha`) no dataset **20 Newsgroups** (~18.000 mensagens reais de Usenet; 11.314 treino / 7.532 teste; 7 parentes / 20 classes):
1. **Classificação Hierárquica** — compara baseline flat vs. classificador hierárquico local por nó (TF-IDF word + char com LinearSVC): acc folha **0.7188** (flat) vs. **0.6953** (hierárquico V3), mas a métrica hierárquica HF mostra quadro mais próximo (0.7668 vs. 0.7516) — a hierarquia ganha em interpretabilidade, não em exact-match.
2. **Clustering Flat vs. Hierárquico** (KMeans, aglomerativo Ward e top-down em 2 níveis, amostra de 3.000 docs seed 42): o **top-down** supera o flat no nível de folha (**Purity 0.398**, **NMI 0.360**, F 0.381) — acima do típico da literatura (NMI 0.25–0.45).

A lição central é que a hierarquia **ajuda mais quando é explorada em níveis** (top-down / cascata), não com um simples corte único; e a classificação não deve ser comparada diretamente ao clustering não supervisionado.

## 2. Contexto e Objetivos

Em muitos problemas as classes não são independentes, e sim estruturadas em árvore (`parente → filho`): no 20 Newsgroups cada rótulo tem a forma `parente.folha` (ex.: `sci.med`), primeira parte é o parente e o grupo é a folha — hierarquia natural de 2 níveis. Os objetivos:

**Parte 1 (classificação):** comparar um modelo flat (20 classes de uma vez) com um classificador hierárquico local por nó (nível 1 prevê o parente; nível 2 prevê a folha dentro do parente) e medir se a estrutura ajuda, prejudica ou é equivalente; descobrir onde a hierarquia erra (nível de pai vs. filho).

**Parte 2 (clustering):** medir o quanto um clustering **flat** (um nível) e um **hierárquico** (grupos aninhados) redescobrem a estrutura real do dataset, tanto no nível de folha (20) quanto de parente (7), sem acesso aos rótulos.

## 3. Fundamentação Teórica (curta)

- **TF-IDF (word + char n-grams)** — representação esparsa; n-grams de caracteres (`char_wb` 2–5) capturam padrões morfológicos complementares. Vetorizadores ajustados **somente no treino** (evita vazamento).
- **LinearSVC** — SVM linear com margem de hiperplano e penalidade L2; junto com n-grams de caracteres generalizou melhor que LogisticRegression com unigramas.
- **Métricas hierárquicas HP/HR/HF** — consideram o rótulo e todos os ancestrais (conjunto aumentado `T* = folha + pai` e `P* = folha + pai`): `HP = |P*∩T*|/|P*|`, `HR = |P*∩T*|/|T*|`, `HF = 2·HP·HR/(HP+HR)`. Capturam erros *parciais* (acerta pai, erra folha ⇒ 0.5) que a exact-match não vê.
- **Clustering externo:** Purity (fração no cluster majoritário), NMI (info mútua normalizada por chance), ARI (índice de Rand ajustado), F cluster. Para validade externa no 20 Newsgroups, valores típicos de NMI para métodos clássicos sobre TF-IDF ficam entre **0.25 e 0.45**.

## 4. Metodologia

### 4.1 Dados

| Item | Classificação | Clustering |
|------|------------------------|------------|
| Base | 20 Newsgroups (`fetch_20newsgroups`) | 20 Newsgroups (conjunto de teste) |
| Treino / teste | **11.314** / **7.532** docs | amostra fixa **3.000** docs (seed 42) |
| Folhas | 20 | 20 |
| Parentes (nível 1) | 7 (`alt`, `comp`, `misc`, `rec`, `sci`, `soc`, `talk`) | 7 |

Removidos cabeçalhos/rodapés/trechos citados (evita vazamento de rótulo em e-mails). Distribuição de treino por parente: alt 480, comp 2.936, misc 585, rec 2.389, sci 2.373, soc 599, talk 1.952. Texto médio ~1.218 caracteres (mediana 491); ~218 documentos vazios pós-limpeza. Matriz final: **7.532 × 189.423 features**.

Estrutura da árvore (dos rótulos):

```
├─ alt    → alt.atheism
├─ comp   → comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware,
│           comp.sys.mac.hardware, comp.windows.x
├─ misc   → misc.forsale
├─ rec    → rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey
├─ sci    → sci.crypt, sci.electronics, sci.med, sci.space
├─ soc    → soc.religion.christian
└─ talk   → talk.politics.guns, talk.politics.mideast, talk.politics.misc,
            talk.religion.misc
```

Nota: `alt`, `misc`, `soc` têm apenas uma folha → no nível 2 a predição é trivial.

### 4.2 Pré-processamento e features

```python
TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=100_000)   # word 1-1 (V1)
```

Versão melhorada (V2/V3) — concatenação **word + char**:

| Grupo | Analisador | n-gram | max_features | min_df | max_df | sublinear_tf |
|-------|-----------|--------|--------------|--------|--------|--------------|
| Palavras | `word` | (1, 1) | 80.000 | 2 | 0.9 | sim |
| Caracteres | `char_wb` | (2, 5) | 150.000 | 2 | 0.9 | sim |

Para o clustering: TF-IDF sublinear word + char concatenado (≈189k) → **SVD de 100 componentes** + normalização, ajustado somente no treino.

### 4.3 Métodos comparados

**Classificação:**
- **Flat** (baseline): um único multiplicador multiclasse (20 classes).
- **Hierárquica local por nó:** nível 1 prediz o pai, nível 2 prediz a folha dentro de cada pai; folhas únicas → predição trivial.
- V1: `LogisticRegression(C=1.0, max_iter=2000)` em TF-IDF word 1-1.
- V2: `LinearSVC(C=0.15)` word+char (C escolhido por varredura em {0.02–2.0}).
- V3: nível 1 tunado via `GridSearchCV`(cv=3) sobre `C ∈ {0.15,0.5,1.0,2.0}` × `class_weight ∈ {None, balanced}` → melhor `{'C':0.5,'class_weight':'balanced'}` (score CV 0.8353); filhos seguem LinearSVC(C=0.15).

**Clustering:**
| # | Estratégia | Detalhe | n_clusters |
|---|-----------|---------|-----------|
| 1 | **Flat** | `KMeans(k=20)` (folha) / `KMeans(k=7)` (parente) | 20 / 7 |
| 2 | **Hieráquico aglomerativo** | `AgglomerativeClustering` (Ward), corta em 7 e 20 | 20 |
| 3 | **Hieráquico top-down (2 níveis)** | `KMeans(k=7)` → re-clusteriza cada grupo (nº de subgrupos = nº de folhas reais); rótulo = par (pai, filho) | 45 |

### 4.4 Avaliação

Classificação: acurácia exact-match de folha, macro-F1, acurácia de pai, acurácia folha-dado-pai-correto, e métricas HP/HR/HF hierárquicas; matrizes de confusão normalizadas no nível pai.
Clustering: Purity, NMI, ARI, F para folha; Purity e NMI para parentes. Amostra fixa seed 42 garante comparação justa e custo O(n²) do aglomerativo.

### 4.5 Reprodução

```bash
# Classificação
pip install scikit-learn numpy pandas matplotlib jupyter nbconvert
python -m nbconvert --to notebook --execute --inplace classificacao_hierarquica.ipynb --ExecutePreprocessor.timeout=1200
# Clustering
python -m nbconvert --to notebook --execute --inplace clustering_flat_vs_hierarquico.ipynb --ExecutePreprocessor.timeout=1800
```

Ou interativo: `jupyter notebook <notebook>.ipynb`. Dataset baixa automaticamente no primeiro uso (cache scikit-learn).

Ambiente real: Windows, Python 3.13.5, pip 25.1.1, scikit-learn 1.9.0, jupyter/nbconvert/nbformat; deps diretas: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `scipy`.

## 5. Resultados

### 5.1 Classificação — V1 (TF-IDF word 1-1; LogisticRegression)

| Métrica | Flat | Hierárquico |
|---------|------|-------------|
| Acurácia folha (exact) | **0.6835** | 0.6516 |
| Macro-F1 folha | 0.6680 | 0.6401 |
| Acurácia no pai | 0.7889 | 0.7740 |
| Acurácia folha dado pai correto | — | 0.8419 |

### 5.2 Classificação — V2 (word+char; LinearSVC C=0.15)

| Métrica | Flat | Hierárquico | Δ Flat | Δ Hier. |
|---------|------|-------------|--------|---------|
| Acurácia folha (exact) | **0.7188** | 0.6889 | +0.0353 (3.53 pp) | +0.0373 (3.73 pp) |
| Macro-F1 folha | 0.7045 | 0.6786 | +0.0365 | +0.0385 |
| Acurácia no pai | 0.8148 | 0.8054 | +0.0259 | +0.0314 |
| Acurácia folha dado pai correto | — | 0.8554 | — | +0.0135 |

Erros V1→V2: 2.624 → **2.343** (test 7.532).

### 5.3 Classificação — V3 (nível 1 tunado: balanced + C=0.5)

| Métrica | Flat | Hierárquico (V3) |
|---------|------|------------------|
| Acurácia folha (exact) | **0.7188** | 0.6953 |
| Macro-F1 folha | 0.7045 | 0.6848 |
| Acurácia no pai | 0.8148 | **0.8079** |
| Acurácia folha dado pai correto | — | **0.8606** |
| **HF (hierárquica)** | 0.7668 | 0.7516 |

Evolução hierárquica: folha 0.6516 (V1) → 0.6889 (V2) → **0.6953** (V3); pai 0.7740 → 0.8054 → **0.8079**. Erros hierárquico: 2.343 → **2.295** (V3).

**Confusions (hierárquico, V3):**

| Classe verdadeira | Predito | Nº de erros |
|-------------------|---------|-----------|
| `talk.politics.misc` | `talk.politics.guns` | 93 |
| `alt.atheism` | `talk.religion.misc` | 44 |
| `comp.windows.x` | `comp.graphics` | 44 |
| `talk.religion.misc` | `soc.religion.christian` | 43 |
| `soc.religion.christian` | `talk.religion.misc` | 38 |

Classes de política/religião são as mais difíceis (menor F1: `talk.religion.misc` ~0.38, `talk.politics.misc` ~0.48).

### 5.4 Exploração de hiperparâmetros (flat, acc folha)

| Configuração | Acurácia folha |
|--------------|----------------|
| word 1-1 + LogisticRegression (V1) | 0.6835 |
| word 1-1 + LinearSVC(C=0.5) | 0.7010 |
| char_wb 2-5 + LinearSVC(C=0.5) | 0.7080 |
| **char_wb 2-5 + word 1-1 + LinearSVC(C=0.15)** | **0.7188** (escolha final) |

| Configuração do pai | Acurácia pai |
|---------------------|------|
| LinearSVC(C=0.15), sem rebalance | 0.8054 |
| LinearSVC(C=0.5), `class_weight='balanced'` | **0.8079** (escolha final) |

### 5.5 Clustering — nível de folha

| Estratégia | n_clusters | Purity | NMI | ARI | F |
|------------|-----------|--------|-----|-----|---|
| **Flat (KMeans k=20)** | 20 | 0.332 | 0.315 | 0.144 | 0.374 |
| Hieráquico aglomerativo (corte 20) | 20 | 0.289 | 0.280 | 0.102 | 0.335 |
| **Hieráquico top-down (2 níveis)** | 45 | **0.398** | **0.360** | 0.102 | **0.381** |

### 5.6 Clustering — nível de pai

| Estratégia | Purity | NMI |
|-----------|--------|-----|
| **Flat (KMeans k=7)** | **0.568** | **0.268** |
| Aglomerativo (corte 7) | 0.487 | 0.214 |

Notas: o **top-down gera 45 clusters-folha** (não 20) porque os grupos reais não são perfeitamente separados no nível pai; o aglomerativo tem o mesmo nº de clusters do flat (20), comparados equidade.

## 6. Discussão

**Classificação:** o nível 1 (pai) é o gargalo da hierarquia — quando erra, a folha é perdida. Rebalancear (`class_weight='balanced'`) + `C=0.5` elevou o pai de 0.805→0.808 e, em cadeia, a folha de 0.689→0.695. O flat continua vencendo o exact-match (0.7188) porque erros se acumulam ao descer a árvore. Porém a HF hierárquica mostra quadro mais justo (Flat 0.7668 vs 0.7516): acertar o pai mas errar folha conta 0.5; quando o pai acerta, o filho acerta a folha em **86.1%**. Vantagens da hierarquia: **interpretabilidade** (onde o erro ocorre: pai ou filho), **escalabilidade** (modelos por nó, útil p/ milhares de classes), robustez em problemas com muitas classes.

**Clustering**
- Flat KMeans: baseline razoável (ARI 0.144), mas aglomerativo com um único corte (20) fica **abaixo** do flat (NMI 0.280 vs 0.315) — "árvore" não equivale a "hierarquia explorada".
- **Top-down vence no nível de folha** (Purity 0.398, NMI 0.360, F 0.381): ao agrupar primeiro os assuntos e depois os grupos dentro de cada assunto, aproveita a estrutura real `assunto → grupo`.
- No nível de **parente**, o Flat KMeans ainda supera o aglomerativo (Purity 0.568 vs 0.487).
- Quanto ao resultado é bom: clustering **não pode** chegar perto da classificação (acc ≈ 0.72) — não vê rótulo algum; valores típicos de NMI (no 20) são 0.25–0.45; nosso flat (0.315) é esperado, top-down (0.360) e purity (0.398) ficam **acima do esperado**. Purity 0.40/NMI 0.36 = clusters recuperam os grandes tópicos (com palavra, esporte, ciência, religião/política) com misturas consideráveis, já que alguns grupos são quase inseparáveis por vocabulário (`talk.politics.*`, `talk.religion.misc`, `soc.religion.christian`, `alt.atheism`).
- **Limitação comum:** classes de política/religião permanecem as mais confundidas e as minutos nestas células não melhoram a hierarquias.

## 7. Conclusões e Recomendações

- **Representação de features importa**: word + char n-grams + LinearSVC (C=0.15) elevou o flat de ~0.68 para ~0.72.
- **Classificação hierárquica**: mais interpretável e escalável, com ~2.4 pp de perda de exact-match para o flat; a cadeia de erro se concentra no pai — ao reforçar o nível 1 (rebalanceamento + C) a acurácia da folha-subiu e a HF aproximou flat/hierárquico.
- **Clustering top-down (2 níveis) é superior ao flat no nível de folha** (Purity 0.398, NMI 0.360) e **acima da literatura de text clustering no 20 Newsgroups**.
- **Recomendação prática:** para dados com taxonomia conhecida (2+ níveis), o **clustering em cascata (top-down)** é superior a um único KMeans; para classificação, se exact-match é crítica prefira flat — se interpretabilidade e escalabilidade importam, use hierárquica com pai bem treinado.
- **Próximos passos** (class): reformar o nível pai, calibrar via `decision_function`/`predict_proba` ou testar modelos não lineares (pequenos ensembles/redes) para as classes mais confundidas.

## 8. Referências e Arquivos

**Classificação** (`classificacao_hierarquica.ipynb`)
- Notebook principal (executado, contém tabelas, matrizes, gráficos, predições).
- Scripts: `build_notebook.py`, `explore.py` / `explore2.py` (busca de hiperparâmetros), `resultados.txt`.

**Clustering** (`clustering_flat_vs_hierarquico.ipynb`)
- Notebook principal (executado, com dendrograma e heatmap).
- Scripts: `build_clustering.py`, `explore_clust.py`, `resultados_clust.txt`.

Fonte: README raiz do repositório (tabela resumo "Experimentos Hierárquicos — 20 Newsgroups") e documentação scikit-learn do dataset 20 Newsgroups.