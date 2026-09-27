# Reinforcement Learning (Q-Learning) for AutoML

> **Area:** Reinforcement Learning + MLOps
> **Task:** Hyperparameter optimization (AutoML)
> **Primary metric:** F1-Score / MAE (agent reward)
> **Status:** Completed
> **Datasets:** LightGBM on a sentiment dataset (Senti-Pred, 74,000 rows and 100,000 features; LinearSVC) and Sales Forecast (5.6M transactions, 32 variables)

## 1. Abstract

This experiment builds a **Q-Learning Agent from scratch** to replace traditional hyperparameter optimizers (Random Search or Bayesian Optuna). The environment is a real model (LightGBM) whose "actions" change variables such as Learning Rate, Max Depth and Num Leaves; the reward is the model's F1-Score (or MAE). After exploring with an Epsilon-Greedy strategy, the agent learns the "Bellman Equation" and navigates the mathematical space, finding configurations almost instantly. In production, the agent was submitted to two extreme tests: the full Senti-Pred dataset (74,000 rows × 100,000 features with LinearSVC) and a retail forecast with 5.6M transactions (via **Proxy Training**), where it reached **MAE 1.4297** vs. Optuna's **1.4218** — a technical tie in a fraction of the time.

## 2. Context and Objectives

Hyperparameter optimization is normally done by exhaustive, random or Bayesian search (Optuna). This experiment tests the hypothesis that it is possible to teach an **autonomous RL agent** to navigate the hyperparameter space without supervision, recording the Q-Table and convergence in MLOps. The objectives:

- Build a Q-Learning agent from scratch (no external frameworks).
- Validate it on a real problem (LightGBM) with an F1-based reward.
- **Production proof (Senti-Pred Full Scale):** optimize LinearSVC's `C`, `max_iter` and `tolerance` on the full 74,000 × 100k feature dataset, under extremely high computational stress.
- **Big Data proof (Sales Forecast):** release the agent on a dataset of 5.6M transactions with a **Proxy Training** architecture (limits the model to micro-trees), showing the scalability of AI to optimize AI.

## 3. Theoretical Background (brief)

- **Q-Learning** — a temporal-difference method: `Q(s,a) = Q(s,a) + α·(r + γ·max_a'·Q(s',a') − Q(s,a))`. The agent builds a table (Q-Table) that estimates the value of each state-action pair.
- **Exploration vs exploitation** — via **Epsilon-Greedy**: with probability ε the agent explores random actions; afterwards, it executes the best known action.
- **Bellman Equation** — used as the base record of the Q-Table to go from the initial explorer to the expert (convergence).
- **Proxy Training** — to reduce cost in big data, the training model is "blinded" with **n_estimators=50** and **bagging_fraction=0.15** (micro-trees + rotating samples), obtaining the optimal coordinates in a matter of minutes; afterwards a final model is trained with the SAME coordinates on the full dataset.
- **Inverse Reward** — in the forecasting case, the agent only "earned" points if the **MAE dropped** (negative reward for worsening).

## 4. Methodology

### 4.1 Data / environment

| Experiment | Environment | Actions | Reward | Dimension |
|---|---|---|---|---|
| AutoML Q-Learning | real LightGBM | Learning Rate, Max Depth, Num Leaves | F1 improves / penalizes worsening and time | sentiment dataset |
| Senti-Pred Full Scale | LinearSVC | `C`, `max_iter`, `tolerance` | quality metrics (accuracy) under hundreds of fits | 74,000 rows × 100,000 features |
| Sales Forecast (Proxy RL) | LightGBM (Proxy) | 32 time-series variables | **Inverse Reward:** only gains if MAE drops | 5.6M transactions |

### 4.2 Algorithm
- Q-Table recorded via MLflow (convergence trends and saved final table).
- Epsilon-Greedy exploration → exploration of the Bellman space.
- Recording of the best configuration and evaluation of the final model.

### 4.3 Hardware and tracking
- MLflow to log the Q-Table (`q_table_final.npy`) and the convergence curves.
- Analysis safety in the artifacts of `rl_automl_qlearning.ipynb` (Q-Table among the artifact paths).

### 4.4 Reproduction
- `rl_automl_qlearning.ipynb` — base Q-Learning + LightGBM experiment.
- `rl_sentipred_automl.ipynb` — application to Senti-Pred (LinearSVC full scale).
- `../sales-forecast/rl_proxy_sales_full.ipynb` — Proxy Training RL on Sales Forecast (5.6M).
- Artifact: `mlruns/2/<run_id>/artifacts/q_table_final.npy`.

## 5. Results

| Proof | Scenario | Result |
|---|---|---|
| Base (LightGBM) | AutoML Q-Learning + F1 | The agent learns the Bellman equation; converges to the configuration almost instantly after exploration |
| Production (Senti-Pred full) | LinearSVC 74,000×100,000 (C, max_iter, tol) | Scalability: hundreds of hyperplane fits under stress; proof of RL scaling on the full production dataset |
| **Proxy on Big Data** | Sales Forecast 5.6M rows, 32 vars | **MAE 1.4297 (RL) vs. 1.4218 (Bayesian Optuna)** — a technical tie, but in a fraction of the time; `n_estimators=50` and `bagging_fraction=0.15` |

*(real values from the root README; the full production dataset was mapped in minutes.)*

## 6. Discussion

QLearning-Adapt presented surprising results:

1. **Model-free and interpretable:** the Q-Table is interpretable and inspectable, recorded in MLflow.
2. **Sufficiency in high dimensionality:** the Senti-Pred full-scale test (LinearSVC) required continuous hyperplane optimization over hundreds of fits under computational stress — it proved the viability of applying AI to optimize AI at scale.
3. **Proxy Training (an effective approximation technique):** by restricting the model in the proxy (`n_estimators=50`, `bagging_fraction=0.15`) and using Inverse Reward (lowers MAE), the agent navigated 32 time-series variables in minutes, with near-optimal configurations — a 0.008 MAE difference vs. Optuna in real production.
4. **Limitations:** the proxy is not exact — the technical tie (0.008 difference) indicates that final accuracy depends on training on the full dataset with the coordinates found; the reward depends on metrics (it is not open and it is bound to time). Artifacts: final Q-Table associated with `q_table_final.npy`.

## 7. Conclusions and Recommendations

- **Q-Learning is a competitive approximation to Optuna** in the regime studied (MAE 1.4297 vs 1.4218), resulting in a technical tie with **large savings in search time**.
- It works both for **linear models (SVM)** and for **tree models (GBM)** by changing the action space.
- **Proxy / search transfer on micro-trees** is essential for big data (RF195 frames): the RL agent can optimize at low cost without full-training cost.
- **Recommendation:** for pipelines with large datasets, use the RL agent with the proxy and then retrain the final model with the coordinates found.
- Limitations: dependence on the reward from offline metrics and on reproducibility (seed/record). Convergence loci for production.

## 8. References and Files

- `rl_automl_qlearning.ipynb` — base experiment (Q-Learning + LightGBM, MLflow).
- `rl_sentipred_automl.ipynb` — full-scale LinearSVC (Senti-Pred 74k×100k).
- `mlruns/2/cf1bba04d4c448c09402d9500d5492d2/artifacts/q_table_final.npy` — trained Q-Table (MLflow artifact).
- Big Data case: `../sales-forecast/rl_proxy_sales_full.ipynb` (Proxy RL, 5.6M transactions).
- Reference: Sutton & Barto, *Reinforcement Learning: An Introduction* (Q-Learning, Bellman, Epsilon-Greedy); Optuna as the Bayesian search baseline.