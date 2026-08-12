# MLOps Sales-Forecast v2.2 — Produção de ponta a ponta

Pipeline de produção completo sobre o campeão do repo (`sales-forecast`, LightGBM V2.2):
**serving (FastAPI + MLflow registry) → métricas de custo/latência → drift (PSI) → retrain automático (com cooldown) → dashboard vivo.**

## Arquitetura

```
┌──────────────┐      POST /predict       ┌───────────────────────────────┐
│  dashboard   │◄──── GET /metrics        │  FastAPI (mlops/serve.py)      │
│  (HTML vivo) │◄──── GET /recent         │  - carrega Production do MLflow│
└──────────────┘                          │  - loga latência/custo (SQLite)│
                                          └──────────────┬────────────────┘
                                                         │
                                    MLflow registry (experiments/mlruns)
┌──────────────┐    a cada N min        ┌───────────────────────────────┐
│  monitor.py  │──── drift PSI/share ──►│  retrain_trigger.json          │
│  (--auto)    │                        └──────────────┬────────────────┘
└──────┬───────┘                                       │ se drift > limiar
       │ cooldown OK? ────────────────────────────────►▼
       │ fetch: se em cooldown -> skip    ┌───────────────────────┐
       └──────────────────────────────┬──►│ retrain.py → nova run │
                                      │   │ MLflow → Production    │
                                      └───┘ (auto via monitor --auto)
```

## Componentes

| Arquivo | Papel |
|---|---|
| `config.py` | caminhos, limiares (PSI 0.25, share 0.40), custo ($0.0009/1k pred), cooldown de retrain (1800s) |
| `metrics_store.py` | SQLite: predições, drift checks, retrains (`last_retrain_ts`) |
| `model_wrapper.py` | pyfunc do SalesForecasterV2 (dados 2022 + forecasts), com cache dos dados por processo |
| `model_wrapper_local.py` | fallback: carrega o joblib committed se o registry estiver vazio |
| `register_model.py` | retreina o campeão (use_log_target=False) e registra Production |
| `retrain.py` | reusa o pipeline, loga nova run MLflow, promove a versão auto-registrada |
| `serve.py` | FastAPI: `/predict /metrics /recent /drift /health /dashboard` |
| `dashboard.html` | dashboard vivo (polling 5s) |
| `monitor.py` | drift PSI por feature + share-change por categórica; `--auto` executa retrain com cooldown |

## Como rodar

```bash
# 1) Registrar campeão no MLflow (retreina use_log_target=False, ~4 min)
python -m mlops.register_model

# 2) Subir API + dashboard vivo
python -m mlops.serve           # http://localhost:8000/dashboard

# 3) Monitor contínuo com retrain automático
python -m mlops.monitor --loop --auto

# 4) Retrain manual / por gatilho
python -m mlops.retrain --reason manual
```

## Detectando drift (modos do monitor)

```bash
python -m mlops.monitor                    # uma rodada (simulação suave -> tipicamente OK)
python -m mlops.monitor --loop --auto      # contínuo + retrain automático com cooldown
python -m mlops.monitor --auto --dry-run   # mostra se retreinaría, sem retreinar
python -m mlops.monitor --strong           # simulação forte (gatilho garantido p/ demo/testes)
```

### Gatilho automático (cooldown)

- Drift é detectado quando `max_psi > 0.25` **ou** `max_share_change > 0.40`.
- Quando detectado: escreve `retrain_trigger.json` e **o monitor `--auto` chama `retrain(reason="drift")`**.
- **Cooldown**: retrains automáticos não acontecem mais de uma vez a cada `RETRAIN_COOLDOWN_SECONDS` (1800s), evitando feedback loop em drift persistente. O estado é lido do SQLite (`retrain_events`).

## Observações de serviço (latência + precompute)

O forecast completo é essencialmente *batch* (1,47M+ linhas para 2 semanas × 735k combos).
Antes da otimização, `generate_forecasts` re-rodava `feature_engineering` na tabela inteira
a cada semana do horizonte → ~2,5 min por chamada e 495s só no compute do algoritmo.

**Abordagem em 2 camadas** (`model_wrapper.py`):

1. **Recursão vetorizada (live).** `build_forecast_state` pré-computa o estado invariante
   (matriz deslizante de 53 offsets, 10.8s, cacheado) e `forecast_from_state` roda a
   recursão semana a semana 100% vetorizada → **7.3s/2 semanas** (saída idêntica ao
   original; validado com assert_frame_equal em 12k linhas e no dataset completo).
2. **Forecast pre-computado (cache).** No startup, um thread de background roda
   `precompute_forecasts` (forecast completo de `PRECOMPUTE_HORIZON=12` semanas = matriz
   numpy `(735.304 × 12)` em int64). `/predict` com `weeks <= 12` passa a ser um **lookup
   + top_n em numpy** → resposta em **~80–100ms** (incluindo HTTP). Acima do horizonte,
   cai no modo live (ainda ~3.6s/semana). Estado: `GET/POST /precompute`.

Latência observada no endpoint `POST /predict` (top_n=5):

| Métrica | Antes | v9 (precompute) | Ganho |
|---|---|---|---|
| warm /predict (weeks≤12) | 146.4s | **~0.09s** | **~1.6k×** |
| warm /predict (v8, live) | 146.4s | 6.5s | ~22× |
| cold (startup + build 12 sem) | 165.8s | ~2min (só 1×) | — |
| forecast full (2 sem, algoritmo) | 495.7s | 7.3s | 68× |

Custo estimado: $0.0009 / 1k predições; cada chamada registra `n_predictions`,
`latency_ms`, `cost_usd`. O `top_n` reduz o resultado final, mas as features são
computadas para todas as combinações antes do corte (ou pré-computadas).

## Registry: por que v.source ≠ caminho de artefato

No MLflow 3.x, `get_latest_versions(...)` pode retornar `source` como locator
`models:/m-<hash>` (não o caminho `.../mlruns/<exp>/<run>/artifacts/model`). O
`serve.py` carrega via `models:/<name>/<version>`, que resolve na registry e é
independente do estilo do `source`. Caso histórico: um `create_model_version`
manual com `mlflow.get_artifact_uri()` gerava source quebrado em `mlruns/0` —
por isso registramos sempre a **versão auto-registrada pelo `log_model`** e a
promovemos.

## Modelo registrado

Campeão: `use_log_target=False` (val_mae **1.5074** medido sem Optuna). O joblib
commitado (`sales_forecaster_v2_final.joblib`, use_log_target=True, val_mae 2.71)
é a versão descartada no README do repo — registramos o campeão verdadeiro, o
tipo de divergência que um model registry serve para capturar.