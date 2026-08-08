# Recommender Systems — MovieLens e Recomendação Visual

> **Área:** RecSys
> **Tarefa:** Predição de rating (matrix completion) e recomendação top-K / por similaridade
> **Métrica principal:** RMSE (predição de rating)
> **Status:** Concluído
> **Datasets:** MovieLens 100k (100.000 ratings, 943 usuários, 1.682 filmes); dataset próprio de imagens (recomendação visual)

## 1. Resumo

Esta pasta compara **10 abordagens de recomendação** no MovieLens 100k (sparsity 93,7%), combinando o comparativo de 8 paradigmas de `movielens-recsys.ipynb` com o **AutoRec** (item/user) de `movielens-autorec.ipynb`. O **Item-AutoRec** venceu com RMSE 0.9054, seguido por **Two-Tower (0.9297)** e **SVD (0.9352)**; o BPR, apesar do último lugar em RMSE (1.1138), é o adequado para tarefas de ranking. A pasta inclui ainda `image_recommender.ipynb`, um sistema de recomendação por similaridade visual (embeddings ResNet + cosseno), sem métricas de qualidade embutidas (**TBD**).

## 2. Contexto e Objetivos

O experimento busca responder qual paradigma de recomendação prediz melhor ratings num cenário de **alta esparsidade** (93,7% da matriz usuário-item vazia) e qual generalize para produção:
1. **Movielens recsys (8 paradigmas):** heurística de popularidade, similaridade (KNN user/item), fatoração de matrizes (SVD, BPR) e modelos neurais (NCF, Two-Tower), além de Gradient Boosting tabular com features manuais (LightGBM+FE).
2. **AutoRec (2 abordagens adicionais, total 10):** autoencoders para reconstruir vetores parcialmente observados (sedimentam se representação não-linear ml-dr ganha de MF de baixo posto).
3. **Cold-start:** simular um usuário novo (3 ratings) e avaliar a coerência das recomendações.
4. **Recomendação visual:** similaridade de conteúdo (não colaborativa) por embeddings de imagem.

## 3. Fundamentação Teórica (curta)

- **Colaborative filtering:** usa interações usuário-item; sofre de cold-start e sparsity.
- **Matrix Factorization (SVD, `surprise`):** aproxima a matriz por R ≈ U·Vᵀ (100 fatores) + biases, minimizando MSE sobre os ratings observados; implementação Cython rápida.
- **KNN user/item:** similaridade de cosseno entre perfis; degrada sob alta sparsity (matriz de distância dominada por zeros).
- **BPR (MF pairwise):** otimiza ranking via amostragem negativa (um par (pos, neg) por vez); escolha para top-K, não avalia baseline por RMSE.
- **NCF (NeuMF):** concatenação de embeddings 32-d + MLP [64,32,16] + Dropout.
- **Two-Tower (DLRM-style):** torres MLP independentes para usuário e item + produto escalar; pré-computa embeddings de itens e viabiliza recuperação aproximada (ANN) em catálogos massivos.
- **LightGBM + FE:** features tabulares (média, std, contagem por user/item + interações, 8 features, 500 trees/6k folhas).
- **AutoRec (Sedhain et al., 2015):** autoencoder (input→tanh→hidden→sigmoid) que reconstrói vetores esparsos; loss MSE **mascarada** (apenas entradas observadas (`M==1`) geram gradiente); variantes item-based (vetor = notas recebidas do item, dim N_USERS) e user-based (dim N_ITEMS).

## 4. Metodologia

### 4.1 Dados

- **MovieLens 100k:** 100.000 ratings (escala 1–5) de 943 usuários sobre 1.682 filmes; sparsity de **93,7%**. Carregado via arquivo `u.data` do surprise (`~/.surprise_data/ml-100k/ml-100k`).
- Split do comparativo neural: `train_test_split` 80/20 com `random_state=42` (80.000 treino / 20.000 teste) — mesmo split para NCF, LightGBM e Two-Tower.
- **Anti-vazamento (AutoRec):** ratings de teste nunca entram na matriz de treino (zerados na construção de `R_tr`).
- **Imagens (image_recommender):** dataset local de ~30 imagens (dim embedding 2048) usado para indexar/demonstrar.

### 4.2 Pré-processamento

- Normalização dos ratings para **[0,1]** (`rating/5`) para o AutoRec (como no paper); entradas observadas ficam 0.
- Matriz M binária (máscara) — loss apenas sobre observados.
- Na predição de rating/recsys: sem feature engineering excessiva; LightGBM+FE dependência de 8 features manuais.
- Representação visual: extração de embeddings ResNet pré-treinada → normalização L2 → similaridade de cosseno.

### 4.3 Métodos comparados

| Modelo | Paradigma | Estratégia | Parâmetros (~) |
|--------|-----------|-----------|---------------|
| **Popularidade** | Heurística | Média global por item (min 5 ratings) | 0 |
| **KNN User-based** | Similaridade (user) | Cosseno entre usuários | 0 |
| **KNN Item-based** | Similaridade (item) | Cosseno entre itens | 0 |
| **SVD** | MF (MSE) | Fatores latentes 100 + biases, 20 épocas | ~200k |
| **NCF (NeuMF)** | Neural (concat) | Embs 32-d + MLP [64,32,16] + Dropout | ~2,2M |
| **LightGBM + FE** | GB Tabular | 8 features usuario/item + interações, 500 trees | ~6k folhas |
| **BPR** | MF (pairwise) | Fatores 64-d, loss BPR, amostragem negativa | ~165k |
| **Two-Tower** | Neural (dot) | Embs 32-d + towers MLP [64,32] + produto escalar | ~150k |
| **Item-AutoRec** | Autoencoder | MLP N_USERS→hidden(500)→N_USERS, Tanh+Sigmoid, masked MSE | ~500×2 |
| **User-AutoRec** | Autoencoder | MLP N_ITEMS→hidden(500)→N_ITEMS, masked MSE | ~500×2 |

### 4.4 Avaliação

- **Métrica:** RMSE no split 80/20 `random_state=42` (holdout).
- **AutoRec:** masked MSE por época; tuning de `hidden ∈ {200, 500, 800}` com split treino/validação (90/10), rankeando por `val RMSE`; 100 épocas, Adam.
- **Cold-start:** simulação de usuário novo (Star Wars 5, Fargo 4, Shining 3) → lista de recomendações do SVD.
- **Reprodutibilidade:** experimento registrado no MLflow local (`./mlruns`, experiment `MovieLens_AutoRec`).

### 4.5 Reprodução

- Notebooks relativos a esta pasta:
  - `./movielens-recsys.ipynb` — 8 paradigmas base (Popularidade, KNN, SVD, NCF, LightGBM, BPR, Two-Tower)
  - `./movielens-autorec.ipynb` — adiciona Item-/User-AutoRec (9ª e 10ª abordagens), compara os 10 modelos e registra no MLflow
  - `./image_recommender.ipynb` — pipeline de recomendação visual (CLI + demo interativa)
- Dependências: `surprise` (dados MovieLens), PyTorch, LightGBM, pandas, numpy, MLflow.
- Padrão de artefatos: `experiments/artifacts/<experimento>_<timestamp>_<sha>/`.

## 5. Resultados

### 5.1 Rodada (10 modelos) — AutoRec incluído (movielens-autorec.ipynb)

| Modelo | RMSE | Paradigma |
|---|---|---|
| **Item-AutoRec** | **0.9054** | Autoencoder (item) |
| Two-Tower | 0.9297 | Neural (dot) |
| SVD | 0.9352 | MF (MSE) |
| LightGBM+FE | 0.9406 | GB Tabular |
| NCF | 0.9462 | Neural (concat) |
| User-AutoRec | 0.9611 | Autoencoder (user) |
| Popularidade | 1.0171 | Heurística |
| KNN User | 1.0194 | Similaridade |
| KNN Item | 1.0264 | Similaridade |
| BPR | 1.1138 | MF (pairwise) |

Nota: no notebook de origem as 8 abordagens originais seguem o padrão de RMSE relatado (Two-Tower 0.929712, SVD 0.935171, LightGBM+FE 0.940597, NCF 0.946228, Popularidade 1.017112, KNN User 1.019354, KNN Item 1.026430, BPR 1.113827).

**AutoRec (detalhes):**
- Item-based: masked MSE convergente 0.0230 (época 100); **test RMSE 0.9054**; treino **16.0s**.
- User-based: masked MSE 0.0239; **test RMSE 0.9611**; treino **10.7s**.
- Tuning (val RMSE): item `hidden=500` → 0.9065 (pico), `hidden=800` → 0.9080, `hidden=200` → 0.9124; user `hidden=500` → 0.9614, `hidden=200` → 0.9649, `hidden=800` → 0.9665.

**Análise da rodada:**
- Item-AutoRec derrota o Two-Tower por ~0.024 em RMSE, e o SVD por ~0.030 — a representação não-linear comprimida (item-side) supera fatorização de baixo posto e os discípulos neurais.
- User-AutoRec (0.9611) fica atrás de LightGBM (0.9406) e NCF (0.9462), mas à frente de heurísticas; **item-side >> user-side** nesta matriz de alta sparsity (heurística melhora per-espancidade média por item do que por usuário).
- BPR em último (1.1138): RMSE não é métrica justa para pairwise-ranking (avaliar precision/recall@K).

### 5.2 Cold-Start (movielens-recsys.ipynb)

Usuário novo avaliou Star Wars 5, Fargo 4, Shining 3 → SVD recomendou **Empire Strikes Back (4.97)**, **Dr. Strangelove (4.94)**, **Cuckoo's Nest (4.94)** — clássicos bem avaliados de perfil similar, indicando coerência sem treinamento do usuário.

### 5.3 Recomendação Visual (image_recommender.ipynb)

- Pipeline: coleta → extração de embeddings ResNet → L2 normalização → cosseno top-K → saída JSONL.
- Benchmark de indexação observada: **30 imagens indexadas em 4.229 s, dim 2048**.
- **TBD:** sem métricas de qualidade (precision@K, recall@K) — a aplicação demonstrações via CLI/demo, sem avaliação formal de ranking nesta pasta.

## 6. Discussão

- **AutoRec item-based é o novo campeão local:** com MLP mascarada (não-linear) o item representa informação compacta que latent-space SVD de posto baixo não aproveita; o custo é treino ~16s vs ~ 11Cython segundos do SVD — trade-off ainda favorável em dataset pequeno.
- **Two-Tower vs SVD** mantém empate mcase local (~0.005–0.006), mas o Two-Tower habilita ANN/max recovery para produção de milhões de itens (Google/Meta/Pinterest).
- **LightGBM+FE** prova que FE manual (u_std, u_mean) compete com neurais (0.9406), além de ser interpretável (SHAP/feature importance).
- **Sparsity prejudica métodos de similaridade** (KNN ~1.02); modelo é sample também do vitorá do Feature Engineering tabular: árvores ganham menos com FE do que modelos lineares (ver pasta `tabular_regression`).
- **BPR não deve ser avaliado por RMSE** — sua função é top-K (precision/recall@K); incluir medição de ranking é refinância futura.
- **Limitações:** dataset único (100k), sem avaliação de ranking (nDCG/precision@K) nas 8 abordagens, e imagem_recommender sem baseline de qualidade — só demonstração do pipeline.

## 7. Conclusões e Recomendações

- **Predição de rating** em datasets ≤ 100k: **AutoRec item-based** (RMSE 0.9054) ou **SVD** (0.9352) como melhor custo/simplicidade (Cython, segundos).
- **Melhor trade-off produção:** **Two-Tower** para escala (ANN) —o custo ~0.93 de RMSE é o melhor se precisa de recuperação em catálogo massivo.
- **Interpretabilidade:** **LightGBM+m3** com 8 features (u_mean/u_std) é forte (0.9406) e fornece SHAP.
- **Ranking (top-K):** **BPR** (pairwise) — mas deve ser medido por precision/recall@K, Não por RMSE.
- **Cold-start:** híbrido de popularidade + conteúdo até o usuário acumular interações.
- Sugestão: adicionar quanto timesertal/eval de ranking e avaliar o `image_recommender` com precision@K em cada dataset.

## 8. Referências e Arquivos

- Notebooks: `./movielens-recsys.ipynb`, `./movielens-autorec.ipynb`, `./image_recommender.ipynb`.
- Referências: Sedhain et al. (2015) *AutoRec: Autoencoders Meet Collaborative Filtering*; Koren et al. (2009) *Matrix Factorization Techniques for Recommender* (SVD); Rendle et al. (2009) *BPR*; He et al. (2017) *Neural Collaborative Filtering* (NeuMF); Grafer et al. para Two-Tower/DLRM; Harley et al. (2022) recommend visual embeddings (ResNet).
- Documento de referência do grupo: `docs/modelo-academico-readme.md`.