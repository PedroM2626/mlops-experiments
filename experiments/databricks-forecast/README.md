# Databricks AutoML Forecast — Prophet e DeepAR (importados da cloud) e Equivalente Local

> **Área:** Séries Temporais
> **Tarefa:** Previsão (regressão) de vendas em série temporal
> **Métrica principal:** RMSE / sMAPE (local equivalente); cloud — a medir (TBD)
> **Status:** Cloud importada (notebooks auto-gerados), local equivalente concluído
> **Datasets:** `quantity_sales_transactions` (tabela Delta do workspace Databricks); equivalente local usa vendas sintéticas.

## 1. Resumo

Esta pasta contém a **importação dos notebooks de forecast gerados pelo Databricks AutoML** (Prophet e DeepAR) para a tabela `quantity_sales_transactions`, junto com os scripts de pipeline (`pre-processing`, `training`, `tuning`), todos auto-gerados pela plataforma e exportados do workspace cloud, além do script `download_artifacts.py` que baixa as runs do tracking MLflow remoto. **Resultados cloud não documentados: TBD (dependem da reexecução em cluster Serverless).** Para viabilizar experimentação local, o equivalente local `time_series/databricks-forecast-local-equivalent.ipynb` reproduz Prophet, Prophet+Optuna, SARIMA e ETS sobre vendas sintéticas (sMAPE 5,66% com Prophet+Optuna).

## 2. Contexto e Objetivos

Os notebooks originais do Databricks dependem de um **workspace pago e de credenciais**; são auto-gerados pelo AutoML (título "auto-generated notebook") e, para reproduzir os resultados, exigem compute Serverless e o experimento MLflow `3257488039771013`. O objetivo desta pasta é **preservar e documentar o que veio da cloud**, enquanto o projeto em paralelo cria os **equivalentes open-source locais** (ver `../time_series/databricks-forecast-local-equivalent.ipynb`), sem necessidade de cluster Spark nem credenciais.

## 3. Fundamentação Teórica (curta)

- **Databricks AutoML:** Auto-gera pipelines de treino para tabelas Delta, com alvo e `time_col`/`split_col`, retornando notebooks importáveis e um experimento MLflow com trials.
- **Prophet (Meta):** GAM aditivo com tendência (changepoints) e sazonalidade (Fourier), adotado pelo AutoML para forecast simples.
- **DeepAR:** modelo de _deep learning_ probabilístico (gluonts) para previsão de séries múltiplas/agregadas, usado pelo AutoML quando o DataFrame tem grandes grupos.
- **Hyperopt + SparkTrials** no Databricks para tuning e **MLflow** para tracking.
- **Equivalente local open-source:** Prophet + Prophet+Optuna + SARIMA + ETS, todos executáveis em máquina simples.

## 4. Metodologia

### 4.1 Dados
- Tabela **`quantity_sales_transactions`** (Delta do workspace Databricks), agregada por `time_col` e `split_col` (média do `target_col` quando há múltiplos valores por grupo) — conforme os notebooks exportados.
- Equivalente local (docs `databricks-forecast-local-equivalent.ipynb`): vendas sintéticas com sazonalidade semanal, holdout de 14 dias.

### 4.2 Pré-processamento
- Script auto-gerado `...-nb-preprocessing.py`: leitura da tabela via Spark SQL (`spark.table`), agregação por tempo/grupo.
- Notebooks cloud já trazem o passo "Aggregate data by `time_col` and `split_col`", igualando o target quando necessario.

### 4.3 Métodos comparados
| Contexto | Métodos |
|---|---|
| Cloud (Databricks AutoML) | Prophet (notebook auto-gerado), DeepAR (notebook auto-gerado), Hyperopt + SparkTrials |
| Local (open-source) | Prophet baseline, **Prophet+Optuna (50 trials)**, SARIMA, ETS |

### 4.4 Avaliação
- Cloud: trials comparados no experimento MLflow (`3257488039771013`), métricas gravadas pela plataforma — **não lidas/executadas aqui (TBD)**.
- Local: holdout temporal de 14 dias, métricas **RMSE** e **sMAPE**, seed fixa.

### 4.5 Reprodução
```powershell
# 1) Baixar artefatos das runs do MLflow Databricks (requer .env com DATABRICKS_HOST/DATABRICKS_TOKEN na raiz do repo)
python download_artifacts.py
# ATENÇÃO: não commitar credenciais; use as variáveis via .env

# 2) Rodar o pipeline local no notebook (abrir no Jupyter):
time_series/databricks-forecast-local-equivalent.ipynb

# 3) Re-executar opcionalmente:
jupyter nbconvert --to notebook --execute databricks-forecast-local-equivalent.ipynb --inplace
```
Aviso: os notebooks cloud exigem compute Serverless do Databricks; por isso os resultados de cloud ficam **TBD** até reexecução.

Padrão de saída local: `experiments/artifacts/<experimento>_<timestamp>_<sha>/`.

## 5. Resultados

### 5.1 Cloud (Databricks AutoML)
Execução dos notebooks exportados require workspace Databricks — **Resultados: TBD** (não documentar valores não verificados).

### 5.2 Equivalente Local (`../time_series/databricks-forecast-local-equivalent.ipynb`) — vendas sintéticas, holdout de 14 dias:
| Método | RMSE | sMAPE | Tempo |
|---|---|---:|---|
| **Prophet + Optuna (50 trials)** | **7,7510** | **5,66%** | 11,3s |
| Prophet (baseline) | 8,0511 | 6,39% | 0,17s |
| ETS | 9,6679 | 6,51% | 0,29s |
| SARIMA | 9,9620 | 6,71% | 3,11s |

**Vitória:** Prophet+Optuna com sMAPE 5,66%, **melhorando 11,4% sobre o baseline** do Prophet.

> [!TIP] Não inventar valores: qualquer métrica dos pipelines cloud só deve ser preenchida após reexecução real na plataforma ou leitura do experimento MLflow correspondente.

## 6. Discussão

- **Local:** com padrões semanais complexos, Prophet+Optuna vence com margem sobre o Prophet baseline; SARIMA e ETS ficaram menos eficazes, indicando que a estrutura do dia a dia (semanal) favorece modelos capazes de absorver sazonalidade de período fixo com flexibilidade (changepoints).
- **Cloud vs Local:** os notebooks exportados são auto-gerados e parametrizados para a tabela real (`quantity_sales_transactions`); o equivalente local substitui a necessidade de workspace, mas opera sobre série sintética — a comparação fiel de métricas exigiria exportar a tabela e reexecutar ambos.
- **download_artifacts.py:** usa MLflow remoto (`mlflow.set_tracking_uri("databricks")`) contra o experimento `3257488039771013`; depende de `.env` com credenciais — **não commited**. O objetivo é baixar artefatos de cada run para inspeção local.

## 7. Conclusões e Recomendações

- Prophet é a melhor escolha para séries com sazonalidade clara (sMAPE 5,66% local).
- Embora os notebooks cloud existam nesta pasta, sua execução correta é **TBD** — prefira o pipeline local para análise e treino.
- Use `download_artifacts.py` apenas com credenciais via variáveis de ambiente; nunca commitar tokens.
- Em produção, reutilize a estrutura de `nb-preprocessing.py`/`nb-training.py`/`nb-tuning.py` como referência de pipeline (Databricks ou local com Spark).

## 8. Referências e Arquivos

- [`26-01-30-12_17-Prophet-19d52e499b042d483dc0d841414c98e2.ipynb`](26-01-30-12_17-Prophet-19d52e499b042d483dc0d841414c98e2.ipynb) — notebook auto-gerado de treinamento Prophet (Databricks AutoML) para `quantity_sales_transactions`.
- [`26-01-30-12_17-DeepAR-2f0e47487cb46323278f1a345f799b99.ipynb`](26-01-30-12_17-DeepAR-2f0e47487cb46323278f1a345f799b99.ipynb) — notebook auto-gerado de treinamento DeepAR (Databricks AutoML).
- [`quantity_sales_transactions_...-nb-preprocessing.py`](quantity_sales_transactions_1769709234206-6207-nb-preprocessing.py) — pipeline de pré-processamento (Spark, agregação `time_col`/`split_col`).
- [`...-nb-training.py`](quantity_sales_transactions_1769709234206-6207-nb-training.py) — script de treino auto-gerado.
- [`...-nb-tuning.py`](quantity_sales_transactions_1769709234206-6207-nb-tuning.py) — script de tuning (Hyperopt) auto-gerado.
- [`download_artifacts.py`](download_artifacts.py) — download de artefatos do MLflow Databricks (requer `.env` com `DATABRICKS_HOST`/`DATABRICKS_TOKEN`).

**Equivalente local:** [`../time_series/databricks-forecast-local-equivalent.ipynb`](../time_series/databricks-forecast-local-equivalent.ipynb). Veja também a seção "Experimentos Locais — Equivalentes Open-Source" do `README.md` da raiz para a tabela de equivalência (Prophet+Optuna local ↔ Prophet/DeepAR cloud, Optuna ↔ Hyperopt/SparkTrials, MLflow local ↔ MLflow Databricks).

Referências: Taylor & Letham (Prophet, 2018); Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020); documentação do Databricks AutoML (2026).