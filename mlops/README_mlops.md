# MLOps Sales-Forecast v2.2 — End-to-end production

Complete production pipeline on top of the repo champion (`sales-forecast`, LightGBM V2.2):
**serving (FastAPI + MLflow registry) → cost/latency metrics → drift (PSI) → automatic retrain (with cooldown) → live dashboard.**

Production tracking: **SQLite** (`experiments/mlops_tracking.db`, default;
override via env `MLFLOW_TRACKING_URI`). The legacy file store
(`experiments/mlruns`) keeps the history and stays browsable in
`mlflow_ui` / via override. Reason: the file backend will be deprecated in
Feb 2026 and the file registry is already second-class in MLflow 3.x.

## Architecture

```
┌──────────────┐      POST /predict       ┌───────────────────────────────┐
│  dashboard   │◄──── GET /metrics        │  FastAPI (mlops/serve.py)      │
│  (live HTML) │◄──── GET /recent         │  - loads Production from MLflow│
└──────────────┘                          │  - logs latency/cost (SQLite)  │
                                          └──────────────┬────────────────┘
                                                         │
                                    MLflow registry (experiments/mlruns)
┌──────────────┐    every N min         ┌───────────────────────────────┐
│  monitor.py  │──── drift PSI/share ──►│  retrain_trigger.json          │
│  (--auto)    │                        └──────────────┬────────────────┘
└──────┬───────┘                                       │ if drift > threshold
       │ cooldown OK? ────────────────────────────────►▼
       │ fetch: if in cooldown -> skip    ┌───────────────────────┐
       └──────────────────────────────┬──►│ retrain.py → new run  │
                                      │   │ MLflow → Production    │
                                      └───┘ (auto via monitor --auto)
```

## Components

| File | Role |
|---|---|
| `config.py` | paths, thresholds (PSI 0.25, share 0.40), cost ($0.0009/1k pred), retrain cooldown (1800s) |
| `metrics_store.py` | SQLite: predictions, drift checks, retrains (`last_retrain_ts`) |
| `model_wrapper.py` | pyfunc of SalesForecasterV2 (2022 data + forecasts), with a per-process data cache |
| `model_wrapper_local.py` | fallback: loads the committed joblib when the registry is empty |
| `register_model.py` | retrains the champion (use_log_target=False) and registers Production |
| `retrain.py` | reuses the pipeline, logs a new MLflow run, promotes the auto-registered version |
| `serve.py` | FastAPI: `/predict /metrics /recent /drift /health /dashboard` |
| `dashboard.html` | live dashboard (5s polling) |
| `monitor.py` | 2-level drift (Evidently `DataDriftPreset` when installed + PSI/share-change always); `--auto` runs retrain with cooldown |
| `registry.py` | promote/resolve via alias `production` (+ stage fallback); `latest_unstaged_version` picks the version auto-registered by `log_model` |
| `tests/` | `test_monitor.py` (PSI/share/compute_drift/Evidently-safe) + `test_metrics_store.py` (pred→drift→retrain cycle in a tmp SQLite) + `test_registry_alias.py` (alias/stage/fallback in a tmp file store) + `test_serve_lifespan.py` (lifespan without loading data) |

## How to run

```bash
# 1) Register the champion in MLflow (retrains use_log_target=False, ~4 min)
python -m mlops.register_model

# 2) Start the API + live dashboard
python -m mlops.serve           # http://localhost:8000/dashboard

# 3) Continuous monitor with automatic retrain
python -m mlops.monitor --loop --auto

# 4) Manual / triggered retrain
python -m mlops.retrain --reason manual
```

## Detecting drift (monitor modes)

```bash
python -m mlops.monitor                    # one pass (smooth simulation -> typically OK)
python -m mlops.monitor --loop --auto      # continuous + automatic retrain with cooldown
python -m mlops.monitor --auto --dry-run   # shows whether it would retrain, without retraining
python -m mlops.monitor --strong           # strong simulation (trigger guaranteed for demos/tests)
```

### Automatic trigger (cooldown)

- Drift is detected when `max_psi > 0.25` **or** `max_share_change > 0.40`.
- When detected: it writes `retrain_trigger.json` and **the `--auto` monitor calls `retrain(reason="drift")`**.
- **Cooldown**: automatic retrains never happen more than once every `RETRAIN_COOLDOWN_SECONDS` (1800s), avoiding a feedback loop under persistent drift. The state is read from SQLite (`retrain_events`).

## Service notes (latency + precompute)

The full forecast is essentially *batch* (1.47M+ rows for 2 weeks × 735k combos).
Before the optimization, `generate_forecasts` re-ran `feature_engineering` on the whole table
for every week of the horizon → ~2.5 min per call and 495s on algorithm compute alone.

**Two-layer approach** (`model_wrapper.py`):

1. **Vectorized recursion (live).** `build_forecast_state` pre-computes the invariant state
   (sliding matrix of 53 offsets, 10.8s, cached) and `forecast_from_state` runs the
   week-by-week recursion 100% vectorized → **7.3s/2 weeks** (output identical to the
   original; validated with assert_frame_equal on 12k rows and on the full dataset).
2. **Pre-computed forecast (cache).** At startup, a background thread runs
   `precompute_forecasts` (full forecast of `PRECOMPUTE_HORIZON=12` weeks = numpy matrix
   `(735,304 × 12)` in int64). `/predict` with `weeks <= 12` becomes a **lookup
   + top_n in numpy** → response in **~80–100ms** (including HTTP). Above the horizon,
   it falls back to live mode (still ~3.6s/week). State: `GET/POST /precompute`.

Latency observed on the `POST /predict` endpoint (top_n=5):

| Metric | Before | v9 (precompute) | Gain |
|---|---|---|---|
| warm /predict (weeks≤12) | 146.4s | **~0.09s** | **~1.6k×** |
| warm /predict (v8, live) | 146.4s | 6.5s | ~22× |
| cold (startup + build 12 weeks) | 165.8s | ~2min (only 1×) | — |
| full forecast (2 weeks, algorithm) | 495.7s | 7.3s | 68× |

Estimated cost: $0.0009 / 1k predictions; each call records `n_predictions`,
`latency_ms`, `cost_usd`. `top_n` shrinks the final result, but features are
computed for all combinations before the cut (or pre-computed).

## Registry: alias `production` (primary) + stage (fallback)

Production resolves via `models:/sales_forecaster_v22@production` (`mlops/registry.py`).
`register_model`/`retrain` point the alias and still try the `Production` stage
for compatibility with older deploys; serving accepts both (alias
first, stage second, joblib last). Reason: `get_latest_versions` and
`transition_model_version_stage` have been deprecated since MLflow 2.9 and the
stages will be removed in a future major — the alias is the supported path.

Historical note (MLflow 3.x): `get_latest_versions(...)` could return `source`
as a `models:/m-<hash>` locator (not the `.../mlruns/<exp>/...` path). The
`serve.py` always loaded via `models:/<name>/<version>` or `@alias`, which
resolves in the registry and is independent of the `source` style.

## Registered model

Champion: `use_log_target=False` (val_mae **1.5074** measured without Optuna). The committed
joblib (`sales_forecaster_v2_final.joblib`, use_log_target=True, val_mae 2.71)
is the version discarded in the repo README — we registered the true champion, the
kind of divergence a model registry exists to capture.