# Reinforcement Learning (Q-Learning) para AutoML

> **Área:** Reinforcement Learning + MLOps
> **Tarefa:** Otimização de Hiperparâmetros (AutoML)
> **Métrica principal:** F1-Score / MAE (recompensa do agente)
> **Status:** Concluído
> **Datasets:** LightGBM sobre dataset de sentimento (Senti-Pred, 74.000 linhas e 100.000 features; LinearSVC) e Sales Forecast (5,6M transações, 32 variáveis)

## 1. Resumo

Este experimento constrói um **Agente de Q-Learning do zero** para substituir otimizadores tradicionais de hiperparâmetros (Random Search ou Optuna Bayesiano). O ambiente é um modelo real (LightGBM) cujas "ações" alteram variáveis como Learning Rate, Max Depth e Num Leaves; a recompensa é o F1-Score (ou MAE) do modelo. Após explorar com estratégia Epsilon-Greedy, o agente aprende a "Equação de Bellman" e navega pelo espaço matemático encontrando configurações quase instantaneamente. Em produção, o agente foi submetido a dois testes extremos: o dataset completo do Senti-Pred (74.000 linhas × 100.000 features com LinearSVC) e um forecast de varejo com 5,6M transações (via **Proxy Training**), onde atingiu **MAE 1.4297** vs. **1.4218** do Optuna — empate técnico em uma fração do tempo.

## 2. Contexto e Objetivos

A otimização de hiperparâmetros é normalmente feita por busca exaustiva, aleatória ou Bayesiana (Optuna). Este experimento testa a hipótese de que é possível ensinar um **agente RL autônomo** a navegar pelo espaço de hiperparâmetros sem supervisão, registrando Q-Table e converm-graft em MLOps. Os objetivos:

- Construir um agente Q-Learning do zero (sem frameworks externos).
- Validá-lo em um problema real (LightGBM) com recompensa baseada em F1.
- **Prova em produção (Senti-Pred Full Scale):** otimizar `C`, `max_iter` e `tolerance` do LinearSVC no dataset completo 74.000 × 100k features, sob altíssimo estresse computacional.
- **Prova de Big Data (Sales Forecast):** livrar o agente em base de 5,6M de transações com arquitetura de **Proxy Training** (limita the model to micro-árvores), mostrando escalabilidade de IA para otimizar IA.

## 3. Fundamentação Teórica (curta)

- **Q-Learning** — método de temporal difference: `Q(s,a) = Q(s,a) + α·(r + γ·max_a'·Q(s',a') − Q(s,a))`. O agente constrói uma tabela (Q-Table) que estima o valor de cada estado-ação.
- **Exploração exploitation** — via **Epsilon-Greedy**: com probabilidade ε o agente explora ações aleatórias; depois, executa a melhor ação conhecida.
- **Equação de Bellman** — utilizada como registro base da Q-Table para ir do explorador inicial ao especialista (convergência).
- **Proxy Training** — para reduzir custo em big data, o modelo de treino é "cega" com **n_estimators=50** e **bagging_fraction=0.15** (micro-árvores + amostras rotativas), obtendo as coordenadas ótimas num tempo de minutos, depois um modelo final é treinado com as MESMAS coordenadas no dataset cheio.
- **Recompensa Inversa** — no caso de previsão, o agente só "ganhava" pontos se o **MAE caísse** (recompensa negativa para piora).

## 4. Metodologia

### 4.1 Dados / ambiente

| Experimento | Ambiente | Ações | Recompensa | Dimensão |
|---|---|---|---|---|
| AutoML Q-Learning | LightGBM real | Learning Rate, Max Depth, Num Leaves | F1 melhora / penaliza piora e tempo | dataset de sentimentos |
| Senti-Pred Full Scale | LinearSVC | `C`, `max_iter`, `tolerance` | quality metrics (acurácia) sob milho de fits | 74.000 linhas × 100.000 features |
| Sales Forecast (Proxy RL) | LightGBM (Proxy) | 32 variáveis temporais | **Recompensa Inversa:** só ganha se MAE cair | 5,6M transações |

### 4.2 Algoritmo
- Q-Table registrada via MLflow (trends de convergência e tabela final salva).
- Exploração Epsilon-Greedy → exploração do espaço do Bellman.
- Registro da melhor configuração e avaliação do modelo final.

### 4.3 Hardware e tracking
- MLflow para log da Q-Table (`q_table_final.npy`) e curvas de convergência.
- Segurança de análise nos artefatos de `rl_automl_qlearning.ipynb` (Q-Table no paths de artefato).

### 4.4 Reprodução
- `rl_automl_qlearning.ipynb` — experimento base Q-Learning + LightGBM.
- `rl_sentipred_automl.ipynb` — aplicação ao Senti-Pred (LinearSVC full scale).
- `sales-forecast/rl_proxy_sales_full.ipynb` — Proxy Training RL no Sales Forecast (5,6M).
- Artefato: `mlruns/2/<run_id>/artifacts/q_table_final.npy`.

## 5. Resultados

| Prova | Cenário | Resultado |
|---|---|---|
| Base (LightGBM) | AutoML Q-Learning + F1 | Agente aprende a Bellman; converge para a configuração quase instantamente após exploração |
| Produção (Senti-Pred full) | LinearSVC 74.000×100.000 (C, max_iter, tol) | Escalabilidade: centenaas de fits de hiperplano sob estresse; prova de RL escalando o dataset completo de produção |
| **Proxy em Big Data** | Sales Forecast 5,6M linhas, 32 vars | **MAE 1.4297 (RL) vs. 1.4218 (Optuna Bayesiano)** — empate técnico, mas em uma fração do tempo; `n_estimators=50` e `bagging_fraction=0.15` |

*(valores reais do README raiz; o dataset completo de produção foi mapeado em minutos.)*

## 6. Discussão

O QLearning-Adapt apresentou resultados surpreendentes:

1. **Modelo-livre e interpretável:** a Q-Table é interpretável e inspecionável, registrada no MLflow.
2. **Suffices em alta dimensionalidade:** o teste full-scale do Senti-Pred (LinearSVC) exigiu otimização contínua de Hyperplane em fit de centenas de vezes sob estresse computacional — provou a viabilidade de aplicar IA para otimizar IA em escala.
3. **Proxy Training (é uma técnica de aproximação eficaz):** Ao restringir o modelo no Proxy (`n_estimators=50`, `bagging_fraction=0.15`) e usar Recompensa Inversa (reduz MAE), o agente navegou 32 variáveis temporais em minutos, com configurações quase ótimas — difextern 0.008 de MAE vs. Optuna na produção real.
4. **Limitações:** proxy não é exato — o empate técnico (0.008 de diferença) indica que a precisão final depende do treinamento no dataset cheio com as coordenadas achadas; a recompensa depende de métricas (não é open e ao tempo). Artefatos: Q-Table final associado `q_table_final.npy`.

## 7. Conclusões e Recomendações

- O **Q-Learning é uma aproximação competitiva ao Optuna** no regime estudado (MAE 1.4297 vs 1.4218), resultando em empate técnico com **grande economia de tempo de busca**.
- Funciona tanto para **modelos lineares (SVM)** quanto para **modelos em árvores (GBM)** modificando o espaço de ações.
- A **Proxy / transferência de busca em micro-árvores** é essencial para big data (RF195 frames ele): equipeRL pode otimiz economâ sem custo de treino cheio.
- **Recomendação:** para pipelines com grandes datasets, usar o agente RL com proxy e depois re-aprender o modelo final com as coordenadas achadas.
- Limitações: dependência da recompensa de métricas offline e da reparaabilidade (seed/record). Loci de conver para produção.

## 8. Referências e Arquivos

- `rl_automl_qlearning.ipynb` — experimento base (Q-Learning + LightGBM, MLflow).
- `rl_sentipred_automl.ipynb` — full-scale LinearSVC (Senti-Pred 74k×100k).
- `mlruns/2/cf1bba04d4c448c09402d9500d5492d2/artifacts/q_table_final.npy` — Q-Table treinada (artefato MLflow).
- Caso Big Data: `../sales-forecast/rl_proxy_sales_full.ipynb` (Proxy RL, 5,6M transações).
- Referência: Sutton & Barto, *Reinforcement Learning: An Introduction* (Q-Learning, Bellman, Epsilon-Greedy); Optuna como baseline de busca Bayesiana.