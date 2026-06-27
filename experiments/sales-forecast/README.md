# Modelo Preditivo de Vendas - Hackathon 2025

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completo-success)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)

Este repositorio contem a solucao completa para o desafio de previsao de vendas do Hackathon 2025. O projeto implementa um pipeline de Machine Learning de ponta para prever a demanda semanal de produtos por ponto de venda, utilizando um modelo Gradient Boosting (LightGBM) meticulosamente otimizado para maxima precisao e robustez.

---

## Objetivo do Projeto

O objetivo principal deste projeto e desenvolver um sistema de previsao de vendas (`forecast`) para as primeiras cinco semanas de 2023, com base no historico de transacoes de 2022. A solucao visa otimizar a reposicao de estoque, minimizando rupturas e excessos, e fornecendo uma base de dados solida para a tomada de decisoes estrategicas da empresa.

---

## Metodologia Aplicada (Arquitetura V2.2 com MLOps)

A solucao foi desenvolvida de forma iterativa, evoluindo de um modelo base para um pipeline sofisticado que incorpora as melhores praticas da industria de Data Science:

1.  **Engenharia de Features Abrangente (32 features):** Foram criadas 32 features a partir dos dados brutos, explorando exaustivamente todas as tabelas dimensionais disponiveis:
    * **Features Categoricas Dimensionais (10):** `pdv`, `sku`, `categoria_pdv`, `premise` (On/Off), `categoria`, `subcategoria`, `tipos`, `label`, `marca`, `fabricante`.
    * **Features Ciclicas e de Calendario (4):** `semana`, `trimestre`, `seno_semana`, `cosseno_semana`.
    * **Features de Lag Temporal (7):** Lags de quantidade em 1, 2, 3, 4, 12 e 52 semanas, e lag do preco medio unitario.
    * **Features de Tendencia (2):** Diferenca de lags consecutivos (`lag_diff_1`) para captura de momentum de curto prazo, e coeficiente de variacao (`coef_variacao_4`) para volatilidade relativa.
    * **Features de Janela Movel (11):** Media, desvio padrao, maximo e minimo moveis em janelas de 4, 12 e 52 semanas, com `min_periods=1` para evitar perda de dados.
    * **Feature de Valor Monetario (1):** `preco_medio_unitario` -- receita bruta dividida pela quantidade vendida, capturando o posicionamento de preco do produto.

2.  **Rastreabilidade MLOps (MLflow):** Todo o ciclo de treinamento e registrado no MLflow, incluindo:
    * Hiperparametros individuais (tipo do modelo, learning rate, num_leaves, etc.).
    * Metricas de performance (MAE, tempo de treinamento, tamanho do dataset).
    * Artefatos (modelo `.joblib`, grafico de feature importance `.png`).

3.  **Otimizacao Bayesiana com Early Pruning (Optuna):** Busca Bayesiana com `MedianPruner` e `LightGBMPruningCallback`. Trials nao promissores sao abortados apos 5 iteracoes, economizando drasticamente tempo computacional. Na V2.2, 20 dos 30 trials foram podados automaticamente, completando a busca em ~6 minutos (vs ~6 minutos do treinamento final).

4.  **Conteinerizacao e Testes Automatizados:** Pipeline empacotado em Docker e coberto por 10 testes unitarios via Pytest, abrangendo: engenharia de features, treinamento, previsao, persistencia (round-trip) e geracao de graficos.

5.  **Estrategia de Submissao Preditiva:** O arquivo final respeita o limite de 1.5 milhao de linhas selecionando as combinacoes (PDV, Produto) com base no maior potencial de vendas futuras previsto pelo proprio modelo otimizado.

---

## Estrutura do Repositorio

O projeto esta organizado da seguinte forma para garantir modularidade e clareza:

```
/
├── artifacts/                  # Modelo treinado (.joblib) e graficos (.png)
├── mlruns/                     # Metadados e logs do MLflow
├── data/
│   ├── raw/                    # Dados brutos de entrada (.parquet)
│   └── processed/              # Previsoes finais geradas (.parquet)
├── scripts/
│   ├── forecaster_class.py     # Classe principal do pipeline (SalesForecasterV2)
│   ├── train.py                # Script de treinamento com Optuna e MLflow
│   └── predict.py              # Script de geracao de previsoes
├── tests/
│   └── test_forecaster.py      # 10 testes automatizados com Pytest
├── Dockerfile                  # Imagem Docker para isolamento de ambiente
└── requirements.txt            # Dependencias com versoes exatas
```

---

## Como Executar o Pipeline

O processo e dividido em duas etapas principais: treinamento e previsao. Execute os scripts a partir do terminal, na pasta raiz do projeto.

**1. Instalar Dependencias:**
```bash
pip install -r requirements.txt
```

**2. Rodar Testes Automatizados:**
```bash
pytest tests/ -v
```

**3. Treinar o Modelo:**
```bash
# Treine o LightGBM com Optuna (30 trials com Pruning)
python scripts/train.py --n_trials 30
```
Ao final, o arquivo `sales_forecaster_v2_final.joblib` e o grafico `feature_importance.png` serao criados na pasta `artifacts/`.

**4. Gerar o Arquivo de Submissao Final:**

* **Para gerar o arquivo de SUBMISSAO (limitado a 1.5M de linhas):**
    ```bash
    python scripts/predict.py
    ```

* **Para gerar a previsao COMPLETA (Opcional):**
    ```bash
    python scripts/predict.py --full_forecast
    ```

**5. (Opcional) Executar via Docker:**
```bash
docker build -t sales-forecaster .
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts sales-forecaster
```

---

## Resultados e Comparativo Academico (V2 vs V2.1 vs V2.2)

O modelo foi avaliado em um conjunto de validacao hold-out temporal (semanas >= 48 de 2022), simulando a previsao de dados futuros desconhecidos. A tabela abaixo documenta a evolucao quantitativa ao longo das tres versoes da arquitetura:

| Metrica / Arquitetura | V2 (Base) | V2.1 (MLOps) | V2.2 (Atual) |
|---|---|---|---|
| **MAE (Loss)** | 2.5769 | 2.2340 | **1.4218** |
| **Reducao relativa do MAE** | -- | -13.3% vs V2 | **-44.8% vs V2** |
| **Reducao incremental** | -- | -- | **-36.3% vs V2.1** |
| **Numero de features** | ~21 | 23 | **32** |
| **Features categoricas** | 2 (pdv, sku) | 5 (+categ, marca, categ_pdv) | **10** (+subcateg, tipos, label, premise, fabricante) |
| **Features de preco** | 0 | 0 | **2** (preco_medio_unitario, lag_1_preco) |
| **Features de tendencia** | 0 | 0 | **2** (lag_diff_1, coef_variacao_4) |
| **n_estimators (final)** | 1000 | 500 | **1000** (com early_stopping=50) |
| **Tempo de treinamento** | Horas | ~10 min | **~12 min** |
| **Trials Optuna** | 100 (sem pruning) | 20 (com pruning) | **30** (com pruning, 20 podados) |
| **Tracking MLflow** | Nao | Basico | **Completo** (params, metrics, artifacts, plots) |
| **Testes automatizados** | 0 | 2 | **10** |

### Analise dos Fatores de Melhoria

A reducao de **44.8%** no MAE entre V2 e V2.2 e atribuida aos seguintes fatores, em ordem estimada de impacto:

1. **Features categoricas dimensionais completas** (~40% do ganho): A inclusao de `subcategoria` (42 valores), `tipos` (22 valores), `label` (14 valores), `premise` (On/Off) e `fabricante` (343 valores) permitiu ao LightGBM aprender padroes de demanda especificos por segmento de produto e tipo de ponto de venda. O LightGBM trata categoricas nativamente via histogram-based splitting, evitando a necessidade de one-hot encoding.

2. **Feature de preco medio unitario** (~25% do ganho): A variavel `preco_medio_unitario` (gross_value / quantity) captura o posicionamento de preco do produto, um forte preditor de volume de vendas segundo a teoria de elasticidade-preco da demanda.

3. **Features de tendencia e volatilidade** (~20% do ganho): `lag_diff_1` (momentum de curto prazo) e `coef_variacao_4` (volatilidade relativa) fornecem ao modelo informacao sobre a direcao e estabilidade da demanda recente, complementando as features de nivel (lags e medias moveis).

4. **Espaco de busca expandido do Optuna** (~15% do ganho): A inclusao de `min_split_gain` como hiperparametro e o aumento do intervalo de `n_estimators` (200-800) e `max_depth` (5-15) permitiram ao Optuna encontrar configuracoes mais adequadas ao novo espaco de features.

### O Experimento Frustrado da Transformacao Logaritmica (log1p)

Durante os experimentos rumo a V2.3, testamos a aplicacao da transformacao `log1p` (logaritmo natural de 1 + x) no `target` (`quantidade`), uma tecnica comum para dados extremamente assimetricos (mediana=2, mas max>90000). A ideia era estabilizar o gradiente.

Contudo, ao avaliar o modelo na escala original (via `expm1`), observamos que o MAE saltou drasticamente (piorou) de ~1.42 para **2.7094**. 
**Por que isso acontece?** Ao otimizar `regression_l1` (MAE) sobre `log(y)`, o modelo minimiza essencialmente o *Erro Percentual* (MAPE). Isso faz com que o modelo seja extremamente conservador, punindo desvios em vendas pequenas, mas subestimando as vendas grandes (ex: errar de 1000 para 900 gera um desvio de log pequeno, mas um desvio absoluto gigante de 100). Como a metrica oficial e MAE absoluto, essa transformacao foi testada, mapeada e deliberadamente **excluida** da solucao final. A flag `use_log_target=False` garante que o modelo treine sempre na escala original.

### O Gargalo do CatBoost e o Foco no LightGBM (V2.2)

Foi realizada uma tentativa de escalar o modelo para um Ensemble misturando o **LightGBM** com o **CatBoost** (V2.3). Contudo, a inclusao do CatBoost provou-se inviavel para um pipeline sem aceleracao de hardware (GPU). Devido ao enorme volume de dados (5.6 milhoes de linhas) e a alta cardinalidade de 10 variaveis categoricas (ex: `fabricante` tem 343 categorias unicas), o CatBoost consumiu **mais de 23 GB de RAM** e monopolizou a CPU por mais de 30 horas/core sem sequer terminar o baseline inicial.

Por essa razao, o processo foi abortado em favor do nosso modelo V2.2 (puramente LightGBM). O LightGBM demonstrou uma superioridade assustadora em eficiencia computacional neste projeto, conseguindo processar as mesmas categoricas espessas via *histogram-based splitting* e gerar um modelo 100% otimizado com Optuna em apenas **~12 a 15 minutos**, mantendo o estado da arte e salvando infraestrutura.

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.8+
* **Ambiente de Empacotamento:** Docker & Pytest
* **Bibliotecas Principais:**
    * Pandas 2.0.3 (Manipulacao de Dados Vetorizada)
    * **LightGBM 4.6.0** (Gradient Boosting principal e otimizado)
    * **Optuna 4.5.0 & PruningCallback** (Otimizacao Bayesiana com poda)
    * **MLflow 2.17.2** (Rastreamento, Gestao do Modelo e Governanca MLOps)
    * Scikit-learn 1.3.2 (Metricas)
    * Matplotlib 3.7.2 (Visualizacao de Feature Importance)
    * Joblib 1.4.2 (Serializacao de Artefatos)

---

## Autores - Equipe: BSB Data 01

* **Erick Cardoso Mendes (desenvolvedor)**
* **Pedro Morato Lahoz (relator)**

---

## Licenca

Este projeto esta licenciado sob a Licenca MIT.
