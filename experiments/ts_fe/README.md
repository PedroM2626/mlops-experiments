# Feature Engineering em Séries Temporais: 5 Fases

> **Área:** Séries Temporais
> **Tarefa:** Previsão (regressão) com engenharia de features
> **Métrica principal:** MAE
> **Status:** Concluído
> **Datasets:** Daily Minimum Temperatures (univariado), Beijing PM2.5 (multivariado), séries derivadas para embeddings DL, decomposição sazonal + wavelets.

## 1. Resumo

Jornada de 5 fases em busca do menor MAE na previsão de séries temporais, respondendo: **o que funciona melhor — intuição humana, força bruta estatística (tsfresh) ou algoritmos avançados (Deep Learning)?** O resultado evolui do uni- para o multivariado e para representações de sinais: a vitória final fica com a combinação **Features Manuais + Wavelets (DWT)** com MAE **54,19**; na Fase 5, os **time embeddings (seno/cosseno)** + **Optuna** melhoraram todas as bases isoladas, mas — paradoxalmente — pioraram o modelo híbrido por underfitting induzido pela validação cruzada.

## 2. Contexto e Objetivos

O objetivo era sistematizar a evolução da engenharia de features de séries temporais em um único pipeline comparável, começando do univariado e chegando a representações avançadas (Deep Learning, wavelets, embeddings circulares, HPO). A cada fase, decidimos a representação de features com base no erro absoluto (MAE) obtido em holdout, verificando se "mais features automáticas" realmente ajuda contra a engenharia manual controlada.

## 3. Fundamentação Teórica (curta)

- **tsfresh:** extração automática em massa de features estatísticas de séries; aqui mostrou queda de performance em series curtas.
- **LSTM Autoencoder (PyTorch):** aprende uma representação latente comprimida (vetor de 16 dims) que pode ser usada como feature tabular.
- **Transformada Wavelet Discreta (DWT, `pywt`):** extrair os choques em janelas (7 dias), separando a estrutura de magnitude da tendência/sazonalidade.
- **Embeddings circulares (Seno/Cosseno):** codificar meses/dias de forma que a semana 52 seja próxima da semana 1.
- **Optuna (TPE)** + `TimeSeriesSplit` (validação cross-temporal) como protocolo de Avaliação.

## 4. Metodologia

### 4.1 Dados
| Fase | Série | Abordagens comparadas |
|---|---|---|
| 1 | Daily Minimum Temperatures (univariado) | tsfresh (automático) vs Manual FE |
| 2 | Beijing PM2.5 (multivariado) | tsfresh (automático, 313 features) vs Manual FE (média móvel) |
| 3 | Séries da Fase 2 + representação latente | Manual, DL Embeddings, Híbrido |
| 4 | Séries da Fase 2 + decomposição | Manual + Trend/Sazonalidade → Wavelet (DWT) |
| 5 | Fase 4 com time embeddings | 4 abordagens (Manual, Decomp. Sazonal, Wavelets, Híbrido Total) + Optuna |

### 4.2 Pré-processamento
- Decomposição de sinal em Trend/Seasonality antes da extração de wavelets.
- Lags, rolling windows (média móvel), embeddings circulares (seno/cosseno) para mês/dia.
- Transformadas DWT sobre janelas de 7 dias para capturar choques.

### 4.3 Métodos comparados (Fase 5)
| Abordagem | Descrição |
|---|---|
| 1. Apenas Manual FE | Features manuais clássicas (lags, média móvel) |
| 2. Apenas Decomp. Sazonal | Trend + sazonalidade separadas, sem wavelets |
| 3. Apenas Wavelets (DWT) | Decomposição sazonal + DWT, sem features manuais |
| 4. Híbrido Total | Manual + Decomp. Sazonal + Wavelets (62 features) |

### 4.4 Avaliação
- Indicador: **MAE** em holdout temporal; validação cruzada `TimeSeriesSplit` (3 folds) no Optuna.
- Optuna executou testes de hiperparâmetros (Random Forest: `n_estimators`, `max_depth`, `min_samples_split`).

### 4.5 Reprodução
Notebooks (na própria pasta, já contêm os outputs):
- `automated_vs_manual_fe_ts.ipynb` → Fase 1
- `multivariate_auto_vs_manual_fe.ipynb` → Fase 2
- `dl_embeddings_fe_ts.ipynb` → Fase 3
- `advanced_signal_fe_ts.ipynb` → Fase 4
- `hpo_time_embeddings_ts.ipynb` → Fase 5

```powershell
# Execução opcional (re-executa o notebook no lugar):
jupyter nbconvert --to notebook --execute hpo_time_embeddings_ts.ipynb --inplace
```

Padrão de saída: `experiments/artifacts/<experimento>_<timestamp>_<sha>/`.

## 5. Resultados

| Fase | Vencedor (Fase) | MAE | Detalhes |
|---|---|---|---|
| 1 | Random Forest + Manual FE | **1,76** | tsfresh: centenas de features, ~30 s, MAE 1,79 |
| 2 | Random Forest + Manual FE | **46,07** | tsfresh: 313 features destruindo performance; média móvel vence |
| 3 | Híbrido (Manual + DL) | **57,24** | LSTM Autoencoder comprime em vetor latente de 16 dims |
| 4 | Híbrido Total (Manual + Wavelets) | **54,19** | DWT extrai choques em janelas de 7 dias |

### Duelo Final (Fase 5 — abordagens com Optuna vs Fase 4)

| Abordagem (Fase 5) | Best Params (Optuna) | MAE Fase 5 (com Optuna) | MAE Fase 4 (sem Optuna) |
|---|---|---|---|
| **3. Apenas Wavelets (DWT)** | `n_est: 50, depth: 5, min_split: 5` | **56,74** | 57,23 |
| **1. Apenas Manual FE** | `n_est: 200, depth: 5, min_split: 5` | **57,14** | 57,88 |
| **2. Apenas Decomp. Sazonal** | `n_est: 200, depth: 5, min_split: 4` | **59,63** | 60,82 |
| **4. Híbrido Total (Manual + Sinais)** | `n_est: 150, depth: 5, min_split: 2` | 55,25 | **54,19** (sem HPO vence) |

> [!WARNING]
> A otimização ajudou os modelos simples a não sofrer overfitting; na validação cruzada do Optuna, porém, impôs regularização dura ao modelo Híbrido, que passou a sofrer underfitting no teste final.

## 6. Discussão

- **tsfresh não é bala de prata:** produz centenas de features (313 no PM2.5), explode a dimensão e piora o MAE (1,79 vs 1,76; destruição completa no PM2.5). Features automáticas sem seleção regularizada não ajudam séries curtas e degradam a informação útil.
- **Deep Learning e wavelets:** o embedding latente do LSTM Autoencoder (16 dims) é eficaz mas inferior à representação de sinais explícitos; o DWT (54,19) superou toda tática anterior, capturando os choques em janelas de 7 dias que features estáticas perdem.
- **Paradoxo da validação cruzada:** o Optuna escolheu `max_depth = 5` para abaixar o erro médio nos 3-folds, o que é suficiente para as bases de ~20 features, mas provoca underfitting no Híbrido de 62 features, que precisa de árvores mais profundas para relacionar média móvel e choque da wavelet (perdeu 54,19 → 55,25).
- **Time embeddings + HPO salvam os modelos simples:** representações seno/cosseno + Optuna melhoraram todas as bases isoladas (Wavelet 57,23→56,74; Manual 57,88→57,14), provando que a regularização (o `max_depth` escolhido) evita decorar o passado e melhora a generalização.

## 7. Conclusões e Recomendações

- **Use wavelets (DWT) + time embeddings (seno/cosseno) como coração da engenharia de features em séries temporais**; foi a coroação do projeto. Ciência de sinais > algoritmos de "caixa preta".
- **Não dependa do tsfresh como extração automática** sem controle de dimensionalidade/regularização.
- **Se for treinar o Híbrido Total (62 features), não restrinja `max_depth`** (nem use `min_samples_split` muito alto) — ou use **muito mais de 10 trials de Optuna** para que o otimizador descubra que a complexidade da base exige árvores mais profundas.
- Para bases simples, Optuna + time embeddings servem bem e garantem boa generalização no futuro.

## 8. Referências e Arquivos

- [`automated_vs_manual_fe_ts.ipynb`](automated_vs_manual_fe_ts.ipynb) — Fase 1 (univariado).
- [`multivariate_auto_vs_manual_fe.ipynb`](multivariate_auto_vs_manual_fe.ipynb) — Fase 2 (PM2.5).
- [`dl_embeddings_fe_ts.ipynb`](dl_embeddings_fe_ts.ipynb) — Fase 3 (LSTM Autoencoder).
- [`advanced_signal_fe_ts.ipynb`](advanced_signal_fe_ts.ipynb) — Fase 4 (DWT).
- [`hpo_time_embeddings_ts.ipynb`](hpo_time_embeddings_ts.ipynb) — Fase 5 (Optuna + time embeddings).

Referências: Christ et al., "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh)", 2018 (breve); planejados: uso de `pywt` (wavelets).