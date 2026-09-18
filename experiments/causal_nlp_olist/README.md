# Causal ML em NLP com Dados Reais — Efeito do Atraso na Entrega sobre o Sentimento do Cliente (Olist)

> **Área:** Inferência Causal + NLP
> **Tarefa:** Estimação de efeito causal (ATE/CATE) de tratamento binário sobre texto
> **Métrica principal:** ATE em diferença de risco (pontos percentuais de P(sentimento negativo))
> **Status:** Concluído
> **Dataset:** Brazilian E-Commerce Public Dataset by Olist (2016–2018, ~100 mil pedidos, licença CC BY-NC-SA 4.0) — 7 tabelas linkadas; espelho público `github.com/Mylinear/Brazilian_E_Commerce_Public_Dataset_by_Olist` (download automático pelo notebook).

## 1. Resumo

Pergunta **causal** (não preditiva): entregar após a data estimada **causa** sentimento negativo no texto da avaliação? Formalizamos em Resultados Potenciais (Rubin) + DAG (Pearl): tratamento $T$ = atraso (`delay_days > 0`), outcome $Y$ = sentimento negativo extraído do texto por léxico PT-BR (validado contra a nota humana), confundidores $X$ estritamente pré-tratamento (preço, frete, categoria, UF, pagamento, peso, prazo prometido, mês). Em N = 39.068 pedidos com texto, a associação bruta é enorme (+43,5 p.p., RR 3,3×) e **sobrevive a todos os ajustes**: LPM +44,4 p.p., Logit (AME) +32,5 p.p., Matching +44,0, IPW-Hájek +43,3, AIPW +41,7, S-learner +35,9, T-learner +41,8. A árvore causal honesta confirma efeito positivo em **todas** as folhas. Conclusão: o atraso é alavanca causal (não mera correlação) do sentimento negativo textual.

## 2. Contexto e Objetivos

Experimentos de NLP do repositório tratam de **predição** de sentimento; aqui a pergunta é **intervencionista**: se a operação logística eliminar o atraso, quanto cai a probabilidade de review negativo? É o primeiro experimento do repositório inteiramente dedicado a inferência causal sobre dados públicos reais, sem *ground truth* contrafactual — a credibilidade vem de triangulação de métodos, diagnósticos e refutações.

Questões de pesquisa:

- **RQ1:** o contraste naive (+43,5 p.p.) é confundido por preço/categoria/UF? (Sim parcialmente — mas o efeito ajustado permanece ≥ +32 p.p. em todos os estimadores.)
- **RQ2:** o efeito se reproduz em outcomes independentes do léxico (`review_score ≤ 2`, `sent_score` contínuo)? (Sim.)
- **RQ3:** há heterogeneidade (CATE) por categoria, UF, faixa de preço e período? (Sim, moderada.)
- **RQ4:** as refutações (placebo, trimming, definição alternativa de T, sensibilidade a confundidor omitido) derrubam a conclusão? (Não.)

## 3. Fundamentação Teórica (curta)

- **Resultados potenciais (Rubin):** $Y_i = T_iY_i(1) + (1-T_i)Y_i(0)$; alvo $\tau = E[Y(1)-Y(0)]$ em escala de diferença de risco (p.p.). Hipóteses: SUTVA/consistência, ignorabilidade dado $X$, positividade $0 < e(X) < 1$, temporalidade (atraso precede o review — verificado: entrega ≤ review em 88,6% dos casos).
- **Back-door (Pearl):** ajustar por $X$ pré-tratamento bloqueia $T \leftarrow X \to Y$; pós-tratamento (datas de review, tamanho do texto) propositalmente excluídos.
- **Estimadores:** LPM (OLS com SE robusto HC1), Logit com efeito marginal médio (AME), propensity score + Matching 1-NN no logit, IPW-Hájek, AIPW (duplamente robusto, nuisances via RandomForest), S/T-learners (Künzel et al., 2019).
- **Heterogeneidade:** CATE via T-learner e **árvore causal honesta** (Athey & Imbens, 2016) implementada manualmente: pseudo-outcomes DR + split honesto 50/50 (amostra A constrói a estrutura, amostra B estima por folha).
- **Credibilidade sem ground truth:** bootstrap B=200, placebo (T permutado), dummy outcome (`n_palavras`), trimming de propensity, definição alternativa de T (>1 dia) e sensibilidade a confundidor omitido simulado.

## 4. Metodologia

### 4.1 Dados

Sete tabelas Olist linkadas por `order_id`/`customer_id`/`product_id` (reviews ⨝ orders ⨝ itens agregados ⨝ primeiro produto ⨝ tradução de categoria ⨝ clientes ⨝ pagamentos agregados): base merged 99.224 × 25. Distribuição de `review_score`: 5★ 57,8%, 4★ 19,3%, 3★ 8,2%, 2★ 3,2%, 1★ 11,5%. **Amostra de análise:** `order_status = delivered` + `delay_days` conhecido + texto não-vazio → **N = 39.068** (tratados = 4.275, 10,9%).

### 4.2 Construção de T, Y e X

- **T** = `1{delay_days > 0}` (dias entre entrega e estimativa; tolerância zero; sensibilidade com >1 dia no Cap. 8 do notebook).
- **Y_neg** = `1{sent_score < 0}`; `sent_score = (p − n)/max(1, p+n)` via léxico PT-BR de substring (auditável, ~80 termos); outcomes secundários: `sent_score` contínuo e `Y_low = 1{review_score ≤ 2}` (rótulo humano).
- **X (22 colunas após dummies):** `log_price`, `log_freight`, `n_items`, `log_weight`, `installments`, `est_days` (prazo prometido), `purchase_yearmonth_int` (tendência), `cat_group` (top-8 + outros), `uf_group` (top-5 + outros), `pay_group`.

### 4.3 Validação do NLP

Léxico × nota humana: correlação **0,661**; acurácia de `Y_neg` contra `score ≤ 2` = **0,8375**; classificador TF-IDF → `Y_neg` atinge AUC **0,985** (sinal linguístico forte). TF-IDF descritivo mostra o vocabulário do atraso ("não recebi", "ainda não", "atraso", "não chegou") vs. sem atraso ("antes do prazo", "recomendo", "muito bom").

### 4.4 Avaliação e Reprodução

- Seed 42 global; probabilidade via `cross_val_predict` para AUC de propensity; bootstrap não-paramétrico B=200 com estimadores rápidos (logit).
- Hardware da execução de referência: CPU x86-64, Python 3.13, scikit-learn 1.7.1, statsmodels 0.14.5 (17/09/2026).
- Reproduzir:

```bash
cd experiments/causal_nlp_olist
python -m nbconvert --to notebook --execute causal_nlp_olist.ipynb --inplace
```

Os 7 CSVs (~65 MB) são baixados automaticamente para `data/` na primeira execução (espelho público GitHub; licença CC BY-NC-SA 4.0 — uso não-comercial com atribuição). Para dados já existentes, o notebook também procura em `../datasets/olist/`.

## 5. Resultados

### 5.1 Seleção (quem escreve review difere)

| Grupo | Score médio | P(score ≤ 2) | P(atraso) | Preço médio | n |
|---|---|---|---|---|---|
| Sem texto (59,5%) | 4,417 | 0,053 | 0,060 | R$ 130,26 | 57.285 |
| Com texto (40,5%) | 3,773 | 0,238 | 0,109 | R$ 146,03 | 39.068 |

Quem escreve tem mais extremos e mais atraso → estimamos **SATE** (efeito na subpopulação com texto), não PATE.

### 5.2 Efeito causal estimado (outcome primário `Y_neg`)

Execução 17/09/2026, seed 42:

| Estimador | ATE (diferença de risco) |
|---|---|
| Naive (sem ajuste) | 0,4348 |
| LPM ajustado (OLS + HC1) | 0,4441 (IC95% [0,4289; 0,4594]) |
| Logit — efeito marginal médio | 0,3249 (IC95% [0,3155; 0,3343]) |
| Matching 1-NN (propensity) | 0,4399 |
| IPW-Hájek | 0,4332 |
| **AIPW duplamente robusto (RF)** | **0,4175** |
| S-learner (RF) | 0,3591 |
| T-learner (RF) | 0,4179 |

**Replicação independente do léxico:** `Y_low = score ≤ 2` (rótulo humano): naive 0,4878 → AIPW **0,4749**; `sent_score` contínuo: naive −0,7975 → AIPW **−0,7598** (piora de sentimento).

### 5.3 Diagnósticos e refutações

| Checagem | Resultado |
|---|---|
| Propensity AUC (in-sample / CV-5) | 0,687 / 0,683 (discriminação moderada = sem separação perfeita) |
| Overlap (e_hat ∈ [0,02; 0,5]) | 98,2% da amostra |
| SMD médio \|·\| bruto → IPW | 0,085 → 0,024 (Love plot) |
| Bootstrap B=200 (AIPW-logit) | média 0,4367, IC95% [0,4208; 0,4547] — exclui 0 |
| Placebo (T permutado) | +0,0007 (p = 0,922) — nulo como esperado |
| Dummy outcome (`n_palavras`) | +3,36 palavras (14,9 vs 11,5) — efeito pequeno de estilo |
| Trimming e ∈ [0,02; 0,98] / [0,05; 0,95] | IPW 0,4343 / 0,4428 (estável) |
| T alternativo (> 1 dia) | naive 0,4800 (efeito cresce com a definição mais estrita) |
| Confundidor omitido simulado (γ = 0,05) | τ sobe p/ 0,5044 vs corrigido 0,4486 — seria preciso U muito forte p/ anular |

### 5.4 Heterogeneidade (CATE, T-learner)

- **Categoria:** sports_leisure 0,446 > ... > telephony 0,366 — amplitude ~8 p.p. entre categorias.
- **UF:** RJ 0,456 > RS 0,432 > MG 0,419 > SP 0,394 / PR 0,393.
- **Faixa de preço:** Q2 0,432 ≈ Q3 0,430 > Q4 0,409 > Q1 0,401 — efeito em todas as faixas.
- **Árvore causal honesta** (profundidade 3, 7 folhas, split honesto 50/50): efeito **positivo em todas as folhas** (0,302 a 0,467; amplitude 0,1644); média ponderada 0,4228; correlação com T-learner por folha = 0,759. Split raiz: `purchase_yearmonth_int` (período), seguido de `pay_group`/`est_days`/`log_freight`/`n_items`.

## 6. Discussão

- **Associação ≠ confundimento aqui:** em dados com seleção forte (SMD brutos até 0,285 em UF), o ajuste mal desloca o efeito (43,5 → 42–44 p.p. em IPW/Matching) porque o atraso é pouco prevalente (10,9%) e o outcome é extremamente reativo a ele. O Logit-AME (+32,5) e o S-learner (+35,9) são mais conservadores por suavização do modelo de resultado.
- **Atenuação por erro de medida:** o léxico erra de forma plausivelmente não-diferencial (não "vê" a data de entrega), logo o viés esperado é de **atenuação** — o efeito real tende a ser ≥ o estimado. A triangulação com `review_score` (rótulo humano, AIPW +47,5 p.p.) confirma magnitude e direção.
- **Seleção:** SATE ≠ PATE; quem não escreve tem nota média 4,42 e atraso 6% — plausivelmente efeito menor na população total. Extensão natural: IPW de seleção / Heckman.
- **Limitações:** sem ground truth contrafactual; SUTVA aproximada (atraso binário colapsa 1 vs 30 dias; interferência regional possível); confundidores não observados (qualidade do produto, expectativa, greves); agregação por primeiro item em pedidos multi-item; anonimização GoT e período 2016–2018 limitam generalização; tamanho do texto (possível mediador) excluído de X.

## 7. Conclusões e Recomendações

- **Cumprir o prazo estimado é alavanca causal de sentimento:** eliminar o atraso na subpopulação que escreve reviews reduziria a probabilidade de review negativo em ~32–44 pontos percentuais (estimador central AIPW: ~42 p.p.).
- **Priorização operacional:** CATE maior em RJ, categorias sports_leisure/watches_gifts e pedidos não-pagos com cartão; folhas da árvore honesta sugerem período e prazo prometido como moderadores.
- **Para pesquisa:** dose-resposta contínua (GPS) em `delay_days`; Double ML + Causal Forest (EconML) com cross-fitting; BERTimbau para sentimento calibrado por aspecto (atraso vs qualidade vs atendimento); correção de seleção de texto.

## 8. Referências e Arquivos

- Notebook executado: [`./causal_nlp_olist.ipynb`](./causal_nlp_olist.ipynb) (37 células originais + bootstrap de dados; figuras e saídas incluídas).
- Dados: `data/` (ignorado no git; download automático pelo notebook a partir do espelho `github.com/Mylinear/Brazilian_E_Commerce_Public_Dataset_by_Olist`).
- Referências: Rubin (1974); Rosenbaum & Rubin (1983); Pearl (2009); Künzel et al. (2019, metalearners); Athey & Imbens (2016, honest trees); Chernozhukov et al. (2018, Double ML); Austin (2009, balanceamento); Egami et al. (2018, text-as-outcome); Feder et al. (2022, causal inference in NLP); Olist (2018, dataset, CC BY-NC-SA 4.0).
