# Modelo Preditivo de Vendas - Hackathon 2025

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completo-success)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)

Este repositório contém a solução completa para o desafio de previsão de vendas do Hackathon 2025. O projeto implementa um pipeline de Machine Learning de ponta para prever a demanda semanal de produtos por ponto de venda, utilizando um modelo Gradient Boosting (LightGBM) meticulosamente otimizado para máxima precisão e robustez.

---

## Objetivo do Projeto

O objetivo principal deste projeto é desenvolver um sistema de previsão de vendas (`forecast`) para as primeiras cinco semanas de 2023, com base no histórico de transações de 2022. A solução visa otimizar a reposição de estoque, minimizando rupturas e excessos, e fornecendo uma base de dados sólida para a tomada de decisões estratégicas da empresa.

---

## Metodologia Aplicada (Arquitetura V2.2 com MLOps)

A solução foi desenvolvida de forma iterativa, evoluindo de um modelo base para um pipeline sofisticado que incorpora as melhores práticas da indústria de Data Science:

1.  **Engenharia de Features Abrangente (32 features):** Foram criadas 32 features a partir dos dados brutos, explorando exaustivamente todas as tabelas dimensionais disponíveis:
    * **Features Categóricas Dimensionais (10):** `pdv`, `sku`, `categoria_pdv`, `premise` (On/Off), `categoria`, `subcategoria`, `tipos`, `label`, `marca`, `fabricante`.
    * **Features Cíclicas e de Calendário (4):** `semana`, `trimestre`, `seno_semana`, `cosseno_semana`.
    * **Features de Lag Temporal (7):** Lags de quantidade em 1, 2, 3, 4, 12 e 52 semanas, e lag do preço médio unitário.
    * **Features de Tendência (2):** Diferença de lags consecutivos (`lag_diff_1`) para captura de momentum de curto prazo, e coeficiente de variação (`coef_variacao_4`) para volatilidade relativa.
    * **Features de Janela Móvel (11):** Média, desvio padrão, máximo e mínimo móveis em janelas de 4, 12 e 52 semanas, com `min_periods=1` para evitar perda de dados.
    * **Feature de Valor Monetário (1):** `preco_medio_unitario` -- receita bruta dividida pela quantidade vendida, capturando o posicionamento de preço do produto.

2.  **Rastreabilidade MLOps (MLflow):** Todo o ciclo de treinamento é registrado no MLflow, incluindo:
    * Hiperparâmetros individuais (tipo do modelo, learning rate, num_leaves, etc.).
    * Métricas de performance (MAE, tempo de treinamento, tamanho do dataset).
    * Artefatos (modelo `.joblib`, gráfico de feature importance `.png`).

3.  **Otimização Bayesiana com Early Pruning (Optuna):** Busca Bayesiana com `MedianPruner` e `LightGBMPruningCallback`. Trials não promissores são abortados após 5 iterações, economizando drasticamente tempo computacional. Na V2.2, 20 dos 30 trials foram podados automaticamente, completando a busca em ~6 minutos (vs ~6 minutos do treinamento final).

4.  **Conteinerização e Testes Automatizados:** Pipeline empacotado em Docker e coberto por 10 testes unitários via Pytest, abrangendo: engenharia de features, treinamento, previsão, persistência (round-trip) e geração de gráficos.

5.  **Estratégia de Submissão Preditiva:** O arquivo final respeita o limite de 1.5 milhão de linhas selecionando as combinações (PDV, Produto) com base no maior potencial de vendas futuras previsto pelo próprio modelo otimizado.

---

## Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir modularidade e clareza:

```
/
├── artifacts/                  # Modelo treinado (.joblib) e gráficos (.png)
├── mlruns/                     # Metadados e logs do MLflow
├── data/
│   ├── raw/                    # Dados brutos de entrada (.parquet)
│   └── processed/              # Previsões finais geradas (.parquet)
├── scripts/
│   ├── forecaster_class.py     # Classe principal do pipeline (SalesForecasterV2)
│   ├── train.py                # Script de treinamento com Optuna e MLflow
│   └── predict.py              # Script de geração de previsões
├── tests/
│   └── test_forecaster.py      # 10 testes automatizados com Pytest
├── Dockerfile                  # Imagem Docker para isolamento de ambiente
└── requirements.txt            # Dependências com versões exatas
```

---

## Como Executar o Pipeline

O processo é dividido em duas etapas principais: treinamento e previsão. Execute os scripts a partir do terminal, na pasta raiz do projeto.

**1. Instalar Dependências:**
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
Ao final, o arquivo `sales_forecaster_v2_final.joblib` e o gráfico `feature_importance.png` serão criados na pasta `artifacts/`.

**4. Gerar o Arquivo de Submissão Final:**

* **Para gerar o arquivo de SUBMISSÃO (limitado a 1.5M de linhas):**
    ```bash
    python scripts/predict.py
    ```

* **Para gerar a previsão COMPLETA (Opcional):**
    ```bash
    python scripts/predict.py --full_forecast
    ```

**5. (Opcional) Executar via Docker:**
```bash
docker build -t sales-forecaster .
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts sales-forecaster
```

---

## Resultados e Comparativo Acadêmico (V2 vs V2.1 vs V2.2)

O modelo foi avaliado em um conjunto de validação hold-out temporal (semanas >= 48 de 2022), simulando a previsão de dados futuros desconhecidos. A tabela abaixo documenta a evolução quantitativa ao longo das três versões da arquitetura:

| Métrica / Arquitetura | V2 (Base) | V2.1 (MLOps) | V2.2 (Atual) |
|---|---|---|---|
| **MAE (Loss)** | 2.5769 | 2.2340 | **1.4218** |
| **Redução relativa do MAE** | -- | -13.3% vs V2 | **-44.8% vs V2** |
| **Redução incremental** | -- | -- | **-36.3% vs V2.1** |
| **Número de features** | ~21 | 23 | **32** |
| **Features categóricas** | 2 (pdv, sku) | 5 (+categ, marca, categ_pdv) | **10** (+subcateg, tipos, label, premise, fabricante) |
| **Features de preço** | 0 | 0 | **2** (preco_medio_unitario, lag_1_preco) |
| **Features de tendência** | 0 | 0 | **2** (lag_diff_1, coef_variacao_4) |
| **n_estimators (final)** | 1000 | 500 | **1000** (com early_stopping=50) |
| **Tempo de treinamento** | Horas | ~10 min | **~12 min** |
| **Trials Optuna** | 100 (sem pruning) | 20 (com pruning) | **30** (com pruning, 20 podados) |
| **Tracking MLflow** | Não | Básico | **Completo** (params, metrics, artifacts, plots) |
| **Testes automatizados** | 0 | 2 | **10** |

### Análise dos Fatores de Melhoria

A redução de **44.8%** no MAE entre V2 e V2.2 é atribuída aos seguintes fatores, em ordem estimada de impacto:

1. **Features categóricas dimensionais completas** (~40% do ganho): A inclusão de `subcategoria` (42 valores), `tipos` (22 valores), `label` (14 valores), `premise` (On/Off) e `fabricante` (343 valores) permitiu ao LightGBM aprender padrões de demanda específicos por segmento de produto e tipo de ponto de venda. O LightGBM trata categóricas nativamente via histogram-based splitting, evitando a necessidade de one-hot encoding.

2. **Feature de preço médio unitário** (~25% do ganho): A variável `preco_medio_unitario` (gross_value / quantity) captura o posicionamento de preço do produto, um forte preditor de volume de vendas segundo a teoria de elasticidade-preço da demanda.

3. **Features de tendência e volatilidade** (~20% do ganho): `lag_diff_1` (momentum de curto prazo) e `coef_variacao_4` (volatilidade relativa) fornecem ao modelo informação sobre a direção e estabilidade da demanda recente, complementando as features de nível (lags e médias móveis).

4. **Espaço de busca expandido do Optuna** (~15% do ganho): A inclusão de `min_split_gain` como hiperparâmetro e o aumento do intervalo de `n_estimators` (200-800) e `max_depth` (5-15) permitiram ao Optuna encontrar configurações mais adequadas ao novo espaço de features.

### O Experimento Frustrado da Transformação Logarítmica (log1p)

Durante os experimentos rumo à V2.3, testamos a aplicação da transformação `log1p` (logaritmo natural de 1 + x) no `target` (`quantidade`), uma técnica comum para dados extremamente assimétricos (mediana=2, mas max>90000). A ideia era estabilizar o gradiente.

Contudo, ao avaliar o modelo na escala original (via `expm1`), observamos que o MAE saltou drasticamente (piorou) de ~1.42 para **2.7094**. 
**Por que isso acontece?** Ao otimizar `regression_l1` (MAE) sobre `log(y)`, o modelo minimiza essencialmente o *Erro Percentual* (MAPE). Isso faz com que o modelo seja extremamente conservador, punindo desvios em vendas pequenas, mas subestimando as vendas grandes (ex: errar de 1000 para 900 gera um desvio de log pequeno, mas um desvio absoluto gigante de 100). Como a métrica oficial é MAE absoluto, essa transformação foi testada, mapeada e deliberadamente **excluída** da solução final. A flag `use_log_target=False` garante que o modelo treine sempre na escala original.

### O Gargalo do CatBoost e o Foco no LightGBM (V2.2)

Foi realizada uma tentativa de escalar o modelo para um Ensemble misturando o **LightGBM** com o **CatBoost** (V2.3). Contudo, a inclusão do CatBoost provou-se inviável para um pipeline sem aceleração de hardware (GPU). Devido ao enorme volume de dados (5.6 milhões de linhas) e à alta cardinalidade de 10 variáveis categóricas (ex: `fabricante` tem 343 categorias únicas), o CatBoost consumiu **mais de 23 GB de RAM** e monopolizou a CPU por mais de 30 horas/core sem sequer terminar o baseline inicial.

Por essa razão, o processo foi abortado em favor do nosso modelo V2.2 (puramente LightGBM). O LightGBM demonstrou uma superioridade assustadora em eficiência computacional neste projeto, conseguindo processar as mesmas categóricas espessas via *histogram-based splitting* e gerar um modelo 100% otimizado com Optuna em apenas **~12 a 15 minutos**, mantendo o estado da arte e salvando infraestrutura.

### O Embate Arquitetural: Sktime vs Alta Cardinalidade (OOM Crash)

Durante a fase de testes e avaliação de frameworks, conduzimos um experimento rigoroso para comparar a nossa Feature Engineering manual baseada em `pandas.groupby().rolling()` (nativa em C/Cython) contra a solução automatizada `WindowSummarizer` da aclamada biblioteca **`sktime`**.

**O Teste de Estresse (Panel Data):**
O `sktime` foi instanciado utilizando a estrutura de MultiIndex Hierárquico (`['pdv', 'sku', 'semana']`) em nossa base de 5.6 milhões de registros transacionais de 2022. O resultado empírico revelou uma vulnerabilidade crítica da biblioteca para dados de alta cardinalidade:
1. **Out Of Memory (OOM) Crash:** O método de *split-apply-combine* interno do `sktime` multiplica e infla matrizes em memória ao instanciar cada agrupamento temporal. A execução consumiu 100% da RAM (superando limites do Hypervisor e colapsando o Python instantaneamente) tentando processar as centenas de milhares de combinações de lojas e produtos, provando-se **Não-Escalável**.
2. **Avaliação de Tempo:** Sob uma subamostragem artificial severa de **apenas 50.000 linhas**, o `sktime` exigiu ~3 minutos para extrair as features. Extrapolando linearmente (embora a complexidade na memória seja superlinear), a base total de 5.6M exigiria mais de 5,5 horas ininterruptas apenas na etapa de Feature Engineering, contrastando com os poucos minutos da nossa solução em Pandas.
3. **Equivalência Matemática:** Extraindo apenas o "PDV Campeão de Vendas" (uma única loja gerando ~6.554 registros perfeitamente seriais), o embate foi justo. O Sktime foi muito rápido (1.37s) e gerou o mesmo resultado estatístico que o Pandas (MAE de **2.516** no Pandas vs **2.544** no Sktime), provando que o bottleneck é puramente arquitetural (gestão de memória de High-Cardinality Panel Data), e não algorítmico.

**Conclusão Acadêmica:** O `sktime` é o estado da arte para séries temporais univariadas e de baixa cardinalidade. No entanto, para Dataframes Transacionais massivos e hierárquicos (MLOps em produção corporativa), a rotina otimizada de vetores C/Cython do Pandas que arquitetamos na **V2.2** é indiscutivelmente superior e blindada contra gargalos de hardware.

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.8+
* **Ambiente de Empacotamento:** Docker & Pytest
* **Bibliotecas Principais:**
    * Pandas 2.0.3 (Manipulação de Dados Vetorizada)
    * **LightGBM 4.6.0** (Gradient Boosting principal e otimizado)
    * **Optuna 4.5.0 & PruningCallback** (Otimização Bayesiana com poda)
    * **MLflow 2.17.2** (Rastreamento, Gestão do Modelo e Governança MLOps)
    * Scikit-learn 1.3.2 (Métricas)
    * Matplotlib 3.7.2 (Visualização de Feature Importance)
    * Joblib 1.4.2 (Serialização de Artefatos)

---

## Autores - Equipe: BSB Data 01

* **Erick Cardoso Mendes (desenvolvedor)**
* **Pedro Morato Lahoz (relator)**

---

## Licença

Este projeto está licenciado sob a Licença MIT.
