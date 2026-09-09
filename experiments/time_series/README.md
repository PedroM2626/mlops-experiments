# Séries Temporais e Previsão (Forecast)

> **Área:** Séries Temporais
> **Tarefa:** Previsão (regressão), Classificação de direção, Detecção de Anomalias e Destilação de Conhecimento
> **Métrica principal:** MAE (e RMSE/MAPE/sMAPE, F1, Acurácia, F1-macro conforme o experimento)
> **Status:** Concluído
> **Datasets:** temperatura (UCI), vendas semanais por PDV (`fato_vendas` — ~6,5M transações), consumo elétrico horário (MWh), temperatura diária de Melbourne, CO₂ Mauna Loa, Nilo, Sunspots, GunPoint/ArrowHead/ECG5000 (UEA), preço sintético GBM + manchetes (TS+NLP), Produção Elétrica (Watsonx) e vendas sintéticas (Databricks).

## 1. Resumo

Esta pasta agrega **12 experimentos** de previsão e análise de séries temporais: otimização de **Prophet com Optuna**, o confronto **Prophet vs LightGBM** (MAE 1,7344 vs 1,96), a evolução do **sales forecast** V2→V2.1→V2.2 (MAE 1,4218), **destilação de conhecimento** (LSTM→TCN com 103,9% da performance do Teacher), **detecção de anomalias** (Z-Score F1 0,9954), **classificação de séries** em 6 paradigmas, um **benchmark de 4 paradigmas × 4 cenários**, a fusão **TS+NLP** para direcionamento de mercado, a conversão do forecast em **classificação de direção** e **forecasting probabilístico com DeepAR** (GluonTS/PyTorch). Conclusão central: nenhum paradigma domina universalmente — cada família de modelos vence em séries cuja estrutura lhe favorece (SARIMA para séries suaves, ROCKET para classificação, Z-Score/Prophet para anomalias conservadoras); DeepAR não supera baselines em acurácia pontual em séries únicas curtas, mas oferece forecasting probabilístico nativo (100 trajetórias amostradas, intervalos de confiança).

## 2. Contexto e Objetivos

O estudo nasceu da necessidade de responder **qual abordagem prever melhor dados temporais** — estatística clássica (SARIMA/ETS), Machine Learning de árvores (LightGBM) ou Deep Learning (TCN, LSTM) — sob diferentes estruturas de série (tendência, sazonalidade, ruído, quebras estruturais). Os notebooks locais também foram criados como **equivalentes open-source** dos pipelines cloud comerciais do IBM Watsonx AutoAI/autoai-ts-libs e Databricks AutoML, viabilizando experimentação local sem credenciais cloud.

Questões de pesquisa:
- Um modelo aditivo baseado em calendário (Prophet) pode competir com um Gradient Boosting alimentado por lags/rolling?
- Qual paradigma vence por tipo de série (tendência suave, tendência ruidosa, ciclo longo, regime com jumps)?
- Destilação de conhecimento funciona com professores de rede neural e de árvore?
- Como converter um problema de forecast em um problema de classificação de direção com métricas interpretáveis?

## 3. Fundamentação Teórica (curta)

- **Prophet (Meta):** modelo aditivo (GAM) com decomposição em tendência (changepoints), sazonalidade (Fourier) e feriados; otimizável via `changepoint_prior_scale` e `seasonality_prior_scale`.
- **SARIMA/SARIMAX:** modelo estatístico paramétrico que captura autocorrelação e sazonalidade; aqui com ordem fixa (1,1,1) para evitar 12 fits sazonais.
- **TCN (Temporal Convolutional Network):** convolução dilatada em múltiplas escalas, eficaz para períodos longos.
- **LightGBM:** gradient boosting de árvores com lags, rolling windows e calendário (engenharia de features).
- **ROCKET:** 10k kernels de convolução aleatórios + classificador Ridge linear; **DTW + 1-NN**: baseline elastic.
- **Knowledge Distillation:** transferência de *soft targets* suavizados de um Teacher para um Student menor.
- **Diebold–Mariano (DM):** teste estatístico de igualdade de acurácia preditiva (correção small-sample / Newey-West).
- **Detecção de anomalias:** Z-Score, bandas de perda do Prophet, Isolation Forest, Elliptic Envelope, LOF sobre resíduos.

## 4. Metodologia

### 4.1 Dados
- **Forecast:** temperaturas diárias (UCI); vendas semanais por PDV (`sales`), ~6,5M transações de 2022; consumo elétrico horário.
- **Anomalias:** 3.650 dias de temperatura diária de Melbourne, com contaminação simulada de 3% = 109 anomalias reais.
- **Classificação UEA:** GunPoint (2 classes, 200 am), ArrowHead (3 classes, 211 am), ECG5000 (5 classes, 5.000 am).
- **Benchmark 4×4:** Mauna Loa CO₂ (semanal, trend+saz; H=30), Nilo (anual, declínio; H=8), Sunspots (anual, cíclico ~11 anos, H=25), Sintético (regime changes + jumps, H=30).
- **TS+NLP:** preço sintético via Geometric Brownian Motion (drift 12%, volatilidade 25%, 1.260 pregões) + manchetes financeiras.
- **Forecast→Classification:** `fato_vendas.parquet` agregado em série diária, 80/20 temporal (treino até 21/out, teste 22/out–31/dez, 71 dias).

### 4.2 Pré-processamento
- Prophet: tuning bayesiano (Optuna) de `changepoint_prior_scale` e `seasonality_prior_scale`, modo `multiplicative`, guiado por MAE via Time Series Cross-Validation.
- LightGBM: engenharia de features — lags 1/4/52 semanas, rolling windows, features cíclicas seno/cosseno.
- Forecast→Classification: winsorização no P99 (anomalia de 29,4M de venda diária em 11/09/2022; mediana ≈ 72K), target binário `qtd(t+1) > qtd(t)`, features de lag/ momentum/ médias móveis/ calendário.
- Anomalias: técnicas aplicadas ao **resíduo** da decomposição.

### 4.3 Métodos comparados
| Experimento | Métodos |
|---|---|
| Prophet e Optuna (forecast) | Prophet baseline vs Prophet+Optuna |
| Prophet vs LightGBM | LightGBM c/ FE vs Prophet |
| Sales forecast V2→V2.1→V2.2 | LightGBM com ~21 → 23 → 32 features, Optuna com pruning, MLflow (LightGBM+CatBoost tentativa falhou) |
| Destilação | LSTM+Attenção (Teacher, 1,44M) → TCN (Student, 228k); LGBM Deep (1.500 est.) → LGBM Shallow (50 est.) |
| Anomalias | Z-Score, Prophet (intervalo 99,9%), Isolation Forest, Elliptic Envelope, LOF |
| Classificação 6 paradigmas | 1-NN+DTW, ROCKET, InceptionTime, TSFresh+RF, Transformer Encoder, LightGBM+FE |
| Benchmark 4×4 | SARIMA, Prophet, TCN, LightGBM |
| TS+NLP | LightGBM: TS-only, NLP-only, TS+NLP |
| Forecast→Classification | Logística, Random Forest, XGBoost, LightGBM |
| Equivalentes locais | Prophet+Optuna, SARIMA, ETS, Naive (Watsonx/Databricks locais) |
| DeepAR probabilístico | DeepAR (GluonTS/PyTorch) vs SARIMA, Prophet, LightGBM + métricas probabilísticas (Coverage, CRPS) |

### 4.4 Avaliação
- Splits temporais (sem shuffle), 80/20 treino/teste.
- Métricas: MAE, RMSE, MAPE, sMAPE, Acurácia, F1, F1-macro, AUC-ROC, Precision, Recall; DM com correção Newey-West (MSE).
- Benchmark: seed 42 (numpy/torch), H proporcional potência; hardware: Intel i7, 16GB RAM, RTX 4070 Laptop (CUDA 12.1).

### 4.5 Reprodução
Os notebooks já contêm os outputs embutidos. Para re-executar (evite, são análises determinísticas já executadas):

```powershell
# A partir de experiments/
jupyter nbconvert --to notebook --execute time_series/ibm-watsonx-local-timeseries.ipynb --inplace
jupyter nbconvert --to notebook --execute time_series/databricks-forecast-local-equivalent.ipynb --inplace
```

Padrão de saída: `experiments/artifacts/<experimento>_<timestamp>_<sha>/`.

## 5. Resultados

### 5.1 Prophet vs LightGBM (temperaturas diárias, holdout)
| Modelo | MAE |
|---|---|
| **LightGBM (lags+rolling+calendário)** | **1,7344** |
| Prophet | 1,96 |

### 5.2 Evolução do Sales Forecast (vendas semanais por PDV)
| Versão | MAE | Δ (vs V2) | Principais mudanças |
|---|---|---|---|
| V2 (objetivo) | 2,5769 | — | ~21 features, apenas 2 categóricas |
| V2.1 (MLOps) | 2,2340 | −13,3% | 23 features, 5 categóricas, MLflow |
| V2.2 (atual) | **1,4218** | **−44,8%** (−36,3% vs V2.1) | 32 features, 10 categóricas dimensionais, preço/tendência/volatilidade, 32 trials podados, MLflow+Docker+10 testes Pytest |

Frustração: `log1p` no target fez o MAE piorar para **2,7094** (escala log otimiza erro relativo → modelo conservador); ensemble CatBoost consumiu >23GB RAM e >30h sem finalizar.

### 5.3 Destilação de Conhecimento (consumo elétrico horário)
| Abordagem | Teacher | Student-KD | Student sem KD | Resultado |
|---|---|---|---|---|
| Neural (LSTM→TCN) | 893,20 MW (1,44M par.) | **858,72 MW** (228k) | 939,31 MW | Student-KD retém **103,9%** do Teacher |
| Árvores (LGBM 1.500→50 est.) | — | 148,28 MW | **146,13 MW** | Destilação falhou |

O LightGBM simples (50 estimadores) com MAE **146,13 MW** superou a rede LSTM (893,20 MW) por ~6× com custo computacional quase nulo.

### 5.4 Detecção de Anomalias (temperatura Melbourne, 109 anomalias)
| Técnica | F1 | Precision | Recall (anomalias) |
|---|---|---|---|
| **Z-Score (resíduo)** | **0,9954** | 100,0% | 99,1% (108/109, 0 alarmes falsos) |
| Isolation Forest (resíduo) | 0,9863 | 98,2% | 99,1% (108/109, 2 falsos) |
| Elliptic Envelope (resíduo) | 0,9863 | 98,2% | 99,1% (108/109, 2 falsos) |
| Local Outlier Factor (LOF) | 0,0183 | — | 2/109 (108 falsos) |

Obs.: Prophet com intervalo de 99,9% obteve F1 0,9860, Precision 100,0% e Recall 97,2% (106 anomalias corretas, nenhum falso positivo).

### 5.5 Benchmark de Paradigmas: MAE por Dataset (v2) — vencedores
| Dataset | SARIMA | LightGBM | Prophet | TCN | Vencedor |
|---|---|---|---|---|---|
| CO₂ | **0,40** | 0,53 | 1,47 | 0,58 | **SARIMA** |
| Nilo | **95,09** | 110,23 | 137,35 | 101,61 | **SARIMA** |
| Sunspots | 45,76 | 56,05 | 43,62 | **25,02** | **TCN** |
| Sintético | 7,59 | 5,94 | **5,27** | 5,71 | **Prophet** |

Tempo de treino (s): SARIMA **34,87** (CO₂)/79,49 (Sintético); LightGBM 0,48–0,64; **Prophet 0,14–0,45**; TCN 0,16–2,44. Ranking: SARIMA 1º em CO₂ e Nilo; TCN 1º em Sunspots; Prophet 1º no Sintético.

### 5.6 Diebold–Mariano significativo (p<0,05)
Prophet vs TCN: **3/4** datasets significativo; SARIMA vs LightGBM no Sunspots (p<0,001); SARIMA vs Prophet em CO₂ e Sintético; p=1,000 indica erros quase idênticos (ex.: CO₂ SARIMA vs LightGBM 0,40 vs 0,53, mas correlação de erros torna o teste inconclusivo).

### 5.7 Classificação de Séries Temporais (6 paradigmas)
- **GunPoint (2 cls, 200 am):** ROCKET **1,000**/1,000 em 1,2s; Transformer 0,967; 1-NN+DTW 0,917; LightGBM+FE 0,833; TSFresh+RF 0,783; InceptionTime 0,767 (14s).
- **ArrowHead (3 cls, 211 am):** ROCKET **0,953**/F1 0,657 em 1,8s; InceptionTime 0,766; 1-NN+DTW 0,578; TSFresh+RF 0,516; LightGBM+FE 0,484; Transformer colapsou para 0,094.
- **ECG5000 (5 cls, 5.000 am):** ROCKET **0,889**/F1 0,487 em 33,1s; Transformer 0,878; 1-NN+DTW 0,841 (36min!); LightGBM+FE 0,820; InceptionTime 0,751; TSFresh+RF 0,720.

### 5.8 TS + NLP (mercado sintético, 80/20 temporal)
| Modelo | Acurácia | F1 |
|---|---|---|
| NLP-only | **0,730** | 0,683 |
| TS+NLP | 0,718 | **0,720** |
| TS-only | 0,492 | 0,504 |

### 5.8b TS+NLP em dados reais — 8-K × direção D+1 (`run_tsnlp_edgar.py`)

Piloto real (sintético tinha sinal por construção): 179 filings 8-K
(SEC EDGAR, sem auth) de AAPL/MSFT/NVDA/AMZN/META/TSLA/JPM, 2024-01–2026-08;
sentimento FinBERT no corpo do filing (512 tokens); alvo = alta/queda do
ticker no pregão seguinte; 178 eventos válidos, split temporal 70/30
(teste ≈ 54 eventos — IC largo, ler com cautela):

| Modelo | Acurácia | F1 | AUC |
|---|---|---|---|
| TS/logreg | 0,537 | 0,000 (classe única) | 0,500 |
| TS/lgbm | 0,482 | 0,482 | 0,484 |
| NLP/logreg | 0,500 | 0,542 | 0,510 |
| NLP/lgbm | 0,500 | 0,342 | 0,485 |
| TS+NLP/logreg | 0,500 | 0,542 | 0,510 |
| TS+NLP/lgbm | 0,444 | 0,423 | 0,444 |

Tudo ≈ cara-ou-coroa — o oposto do sintético (NLP-only 0,730). Leitura:
no sintético o sinal existia por construção (notícia defasada → retorno);
em filings reais de large-caps, sentimento + momentum de 5 dias não batem o
mercado — consistente com eficiência informacional em D+1. Limites: teste
pequeno (n≈54), FinBERT truncado em 512 tokens, sem controle de horário do
filing (after-close vs intraday). Artefatos:
`experiments/artifacts/tsnlp_edgar_20260908_194545/` (+ `run_tsnlp_real.py`
documenta a tentativa via GDELT, bloqueada por rate-limit 429).

### 5.9 Forecast→Classificação (teste out-of-sample, 71 dias) — maior = melhor
| Modelo | Acurácia | Bal. Acc | Prec | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---:|
| **Logística** | **0,958** | 0,957 | 0,917 | 0,957 | 0,936 | **0,967** |
| Random Forest | 0,944 | 0,936 | 0,913 | 0,913 | 0,913 | 0,952 |
| XGBoost | 0,944 | 0,936 | 0,913 | 0,913 | 0,913 | 0,942 |
| LightGBM | 0,930 | 0,914 | 0,909 | 0,870 | 0,889 | 0,945 |

Baselines: maioria 66,9% | persistência (d+1) 67,7% | mesmo dia da semana passada 42,2%. Top features: `is_weekend`, `dow` e `lag_7` concentram ~59% da importância no Random Forest.

### 5.10 Equivalentes locais (open-source vs cloud)
**Watsonx local** (`ibm-watsonx-local-timeseries.ipynb`, Produção Elétrica, holdout 20 meses, MAPE):
| Método | RMSE | MAPE | Tempo |
|---|---|---:|---:|
| Prophet+Optuna (100 trials) | **3,5583** | 3,90% | 21,6s |
| SARIMA | 3,5648 | 3,90% | 0,91s |
| Prophet (baseline) | 3,6134 | 4,04% | 0,09s |
| ETS | 3,6412 | 4,08% | 0,05s |
| Naive | 19,0495 | 20,55% | 0,00s |

**Databricks local** (`databricks-forecast-local-equivalent.ipynb`, vendas sintéticas, holdout 14 dias, sMAPE):
| Método | RMSE | sMAPE | Tempo |
|---|---|---:|---:|
| Prophet+Optuna (50 trials) | **7,7510** | **5,66%** | 11,3s |
| Prophet (baseline) | 8,0511 | 6,39% | 0,17s |
| ETS | 9,6679 | 6,51% | 0,29s |
| SARIMA | 9,9620 | 6,71% | 3,11s |

### 5.11 DeepAR Probabilístico (4 datasets do benchmark, 100 amostras, CPU)

Re-execução em 08/09/2026 (GluonTS 0.17, seed 42; `deepar-probabilistic-forecast.ipynb`
com outputs atualizados):

**Forecast pontual (MAE):**
| Dataset | SARIMA | Prophet | LightGBM | DeepAR | Vencedor |
|---|---|---|---|---|---|
| CO₂ | 4,27 | **0,61** | 1,19 | 2,01 | Prophet |
| Nile | 123,01 | **120,12** | 127,24 | 137,94 | Prophet |
| Sunspots | 44,90 | — (falhou: `Overflow in int64 addition` em datas anuais) | 16,89 | **15,22** | DeepAR |
| Synthetic | 8,33 | 4,20 | 4,68 | **4,09** | DeepAR |

**Métricas probabilísticas (DeepAR):**
| Dataset | Coverage(90%) | AvgWidth | CRPS |
|---|---:|---:|---:|
| CO₂ | 90,0% | 6,35 | 2,41 |
| Nile | 75,0% | 370,52 | 164,97 |
| Sunspots | 80,0% | 87,16 | 26,43 |
| Synthetic | 93,3% | 18,37 | 6,35 |

DeepAR venceu 2/4 em MAE nesta re-execução (Sunspots, Synthetic) — treino
estocástico varia entre runs (na execução anterior: 0/4). Coverage segue o
padrão: bom em séries semanais longas (90–93%), fraco nas anuais curtas
(75–80%). Custo: 10–54 s por dataset (GPU disponível p/ Lightning; Prophet
0,1–2 s). Conclusão mantida: DeepAR é relevante para forecasting
probabilístico em séries longas/múltiplas, não substitui baselines em séries
únicas curtas — mas com re-treino pode empatar/vencer em MAE pontual.

### 5.12 Recalibração conformal — medida em séries reais (`run_deepar_conformal.py`)

Split-conformal com pool rolling (6 origens; `n_cal_points` abaixo), nominal 90%:

| Dataset | n_cal | Coverage antes | Depois (global) | Depois (por-h) | Largura antes → por-h |
|---|---|---|---|---|---|
| CO₂ | 180 | 1,00 | 1,00 | 1,00 | 8,1 → 13,9 |
| Nile | 48 | 0,375 | 0,75 | 0,75 | 326,7 → 429,2 |
| Sunspots | 150 | 0,12 | 0,96 | **0,92** | 31,2 → 613,0 |
| Synthetic | 90 | 0,30 | 1,00 | 0,77 | 11,6 → 137,2 |

Lições medidas (não só protocolo):
1. **Janela única de H pontos não basta** (primeira tentativa: CO₂ 0,43→0,57) — o pool rolling (48–180 pontos) foi o que destravou a correção.
2. **Por-horizonte ≈ nominal com menos largura que global** (Sunspots 0,92 com 613 vs 0,96 com 671; Synthetic 0,77 com 137 vs 1,00 com 192).
3. **Conformal é band-aid, não cura**: q_global de 27–35× em Sunspots/Synthetic mostra que o σ nativo é inútil nesses regimes — o modelo precisa de revisão, não só de reescala.
4. **Granularidade do teste limita**: Nile H=8 só permite coverages em passos de 1/8 (0,75 = 6/8) — nunca cravará 0,90 exato.
Artefatos: `experiments/artifacts/deepar_conformal_20260908_130049/metrics.json`.

## 6. Discussão

- **Desafio Prophet vs LightGBM:** em ruído diário abrupto, árvores que "leem" os lags imediatos reagem melhor do que equações aditivas baseadas apenas no calendário estático (MAE 1,73 vs 1,96). Categóricas de alta cardinalidade, `preco_medio_unitario` e features de preço concentraram ~65% do ganho no pipeline V2.2.
- **Benchmark:** SARIMA vence 2/4 cenários mesmo com ordem fixa (1,1,1), mas custa 35–80s em séries longas; TCN só vence em ciclos longos (Sunspots), onde 6 blocos dilatados capturam a periodicidade de ~11 anos; LightGBM nunca vence mas é consistente (2º–3º) e bom quando há features exógenas; Prophet vence regime changes porque a changepoint detection absorve os jumps estruturais que quebram a memória do ARIMA.
- **Classificação:** ROCKET domina pela projeção aleatória de alta dimensão (max + ppv de 10k kernels), Ridge linear; DTW é baseline robusto mas impraticável escala (O(N²·L²); 36 min no ECG5000); Transformer colapsa em ArrowHead (0,094) por falta de dados (147 treino × 251 steps); F1 macro baixo nos datasets multiclasse indica confusão na classe minoritária.
- **Destilação:** a transferência de soft targets funciona para modelos densos (Student-KD máxima 103,9% do Teacher), mas falha quando o Teacher é uma árvore, porque o professor overfitta e gera previsões pontuais idênticas ao ground truth, anulando a suavização (146,13 vs 148,28 MW).
- **Anomalias:** métodos estatísticos (Z-Score/Prophet) são conservadores e ideais quando alarmes falsos são caros; Isolation Forest supera em Recall (≈108/109) com calibração de contaminação; LOF falha por densidade espacial em um resíduo 1D agrupado próximo a zero — exige antes uma decomposição em resíduos ou janelas de lags.
- **TS+NLP e Forecast→Classification:** a feição causal defasada (notícia do dia → retorno de amanhã) domina o sintético; em séries reais, TS+NLP tende a superar ambos isolados. Classificar direção converte métricas de erro em F1/AUC interpretáveis, com forte dominância de calendário (fim de semana ≈ 2,8% das vendas diárias).
- **Equivalentes locais:** SARIMA reproduziu o Prophet+Optuna em Produção Elétrica (MAPE 3,90% igual, 24× mais rápido); Prophet+Optuna vence nas vendas sintéticas (sMAPE 5,66%, +11,4% vs baseline), superando SARIMA/ETS em padrões semanais complexos.
- **DeepAR probabilístico:** o modelo de deep learning autoregressivo (LSTM + Student-T) não superou baselines em acurácia pontual em nenhum dos 4 datasets — Prophet venceu 3/4, LightGBM venceu 1/4. O diferencial do DeepAR é o forecasting probabilístico nativo: 100 trajetórias amostradas, intervalos de confiança calibrados (Coverage 90% ≈ 93–100% em séries semanais longas; undercoverage em séries anuais curtas com <350 obs). Custo computacional 50–600× superior aos baselines (CPU). Recomendado para múltiplas séries correlacionadas (cross-learning) ou quando a distribuição preditiva é requisito de negócio.

## 7. Conclusões e Recomendações

- **Forecast estatístico puro:** use SARIMA para séries suaves (ruído <5%); Prophet quando houver quebras estruturais / feriados; TCN para padrões cíclicos longos (10k+ pontos); LightGBM com FE quando a hipótese paramétrica não se sustenta e existam features exógenas (ex.: vendas).
- Embora LightGBM MAE 1,7344 > Prophet 1,96 em temperatura diária, o modelo de vendas passou de 1,7344 → 2,5769 → 2,2340 → 1,4218 via 32 features + HPO bayesiano com pruning; **não** aplicar `log1p` em target de L1/MAPE.
- Preferir Z-Score/Prophet para alertas onerosos; IsolationForest para maximizing recall; evitar LOF sobre a série crua.
- Para classificar séries: ROCKET é baseline rápido e dominante; InceptionTime para >10k amostras; DTW para <200; LightGBM+FE com SHAP quando interpretabilidade importa.
- Em problema de forecast, considere converter para classificação de direção quando a escala natural dificultar a interpretação de `MAE`.
- Os equivalentes open-source substituem Watsonx/Databricks na linha experimental com métricas comparáveis.

## 8. Referências e Arquivos

Notebooks (na própria pasta):
- [`temperature_forecasting_prophet.ipynb`](temperature_forecasting_prophet.ipynb) e [`property-sales-time-series.ipynb`](property-sales-time-series.ipynb) — Prophet e Optuna, sales V2→V2.2.
- [`knowledge_distillation-time_series.ipynb`](knowledge_distillation-time_series.ipynb) — destilação neural/tabular.
- [`exp4_anomaly_detection.ipynb`](exp4_anomaly_detection.ipynb) — 5 técnicas de anomalia.
- [`benchmark-ts-paradigms.ipynb`](benchmark-ts-paradigms.ipynb) — 4 cenários × 4 arquiteturas.
- [`time-series-classification.ipynb`](time-series-classification.ipynb) — 6 paradigmas, datasets UEA.
- [`stock-sentiment-ts-nlp.ipynb`](stock-sentiment-ts-nlp.ipynb) — TS+NLP.
- [`forecast-classification.ipynb`](forecast-classification.ipynb) — previsão de direção.
- [`ibm-watsonx-local-timeseries.ipynb`](ibm-watsonx-local-timeseries.ipynb) — equivalente Watsonx.
- [`databricks-forecast-local-equivalent.ipynb`](databricks-forecast-local-equivalent.ipynb) — equivalente Databricks.
- [`multivariate-time-series-var.ipynb`](multivariate-time-series-var.ipynb) — Vector Autoregression (VAR) e Impulse Response Functions.
- [`hierarchical_forecast.ipynb`](hierarchical_forecast.ipynb) — previsão hierárquica temporal com reconciliação bottom-up.
- [`deepar-probabilistic-forecast.ipynb`](deepar-probabilistic-forecast.ipynb) — DeepAR (GluonTS/PyTorch) probabilístico vs baselines pontuais.
- [`deepar-generative/deepar-generative-futures.ipynb`](deepar-generative/deepar-generative-futures.ipynb) — DeepAR como modelo generativo: 500 trajetórias, cenários, probabilidades.
- `sktime_vs_hybrid_ts.ipynb` — comparação library/custom hybrid TS de referência.

Referências: Taylor & Letham, "Forecasting at Scale" (Prophet, 2018); Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020); Bromet et al., ROCKET (2020), Diebold & Mariano (1995); Hinton et al., Distilling the Knowledge (2015); UEA Archive, GunPoint/ArrowHead/ECG5000.