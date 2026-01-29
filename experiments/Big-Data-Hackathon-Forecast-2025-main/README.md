# 🚀 Modelo Preditivo de Vendas - Hackathon 2025

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completo-success)

Este repositório contém a solução completa para o desafio de previsão de vendas do Hackathon 2025. O projeto implementa um pipeline de Machine Learning de ponta para prever a demanda semanal de produtos por ponto de venda, utilizando um modelo Gradient Boosting (LightGBM) meticulosamente otimizado para máxima precisão e robustez.

---

## 🎯 Objetivo do Projeto

O objetivo principal deste projeto é desenvolver um sistema de previsão de vendas (`forecast`) para as primeiras cinco semanas de 2023, com base no histórico de transações de 2022. A solução visa otimizar a reposição de estoque, minimizando rupturas e excessos, e fornecendo uma base de dados sólida para a tomada de decisões estratégicas da empresa.

---

## 🛠️ Metodologia Aplicada

A solução foi desenvolvida de forma iterativa, evoluindo de um modelo base para um pipeline sofisticado que incorpora as melhores práticas da indústria de Data Science:

1.  **Engenharia de Features Avançada:** Foram criadas mais de 20 features a partir dos dados brutos para capturar padrões complexos, incluindo:
    * **Lags Sazonais:** Vendas de semanas, meses e do ano anterior para informar o modelo sobre o comportamento histórico.
    * **Janelas Móveis:** Média, desvio padrão e máximo de vendas em diferentes janelas de tempo para identificar tendências e volatilidade.
    * **Features Cíclicas:** Decomposição da sazonalidade semanal usando seno e cosseno para um aprendizado mais eficaz.

2.  **Validação Robusta:** Foi implementada uma estratégia de **validação Hold-Out temporal**, separando as últimas semanas de 2022 para avaliar o modelo em um cenário que simula a previsão de dados futuros e desconhecidos, garantindo uma métrica de performance confiável (MAE).

3.  **Otimização de Hiperparâmetros de Alta Precisão:** O passo decisivo para a performance do modelo foi a utilização da biblioteca **Optuna**. Realizamos uma busca Bayesiana exaustiva com **100 iterações (`trials`)** para encontrar a combinação de hiperparâmetros do LightGBM que minimizasse o erro de previsão, resultando em um modelo final altamente especializado e ajustado para este dataset.

4.  **Estratégia de Submissão Preditiva:** O arquivo final respeita o limite de 1.5 milhão de linhas selecionando as combinações (PDV, Produto) com base no **maior potencial de vendas futuras previsto pelo próprio modelo otimizado**, uma abordagem proativa que foca nos produtos de maior impacto.

---

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir modularidade e clareza:

```
/
├── artifacts/              # Pasta para salvar o modelo treinado (.joblib)
├── data/
│   ├── raw/                # Dados brutos de entrada (.parquet)
│   └── processed/          # Previsões finais geradas pelo script (.parquet)
├── scripts/
    ├── forecaster_class.py     # Arquivo contendo a classe principal do pipeline (SalesForecasterV2)
    ├── train.py                # Script para treinar o modelo LightGBM com Optuna
    └── predict.py              # Script para gerar a previsão final usando o modelo treinado

```

---

## ▶️ Como Executar o Pipeline

* Salve todas as pastas deste repositório em uma única, com o nome de sua preferência.

O processo é dividido em duas etapas principais: treinamento e previsão. Execute os scripts a partir do terminal, na pasta raiz do projeto.

**1. Treinar o Modelo:**
   *Este processo é computacionalmente intensivo e pode levar várias horas.*
   
   ```bash
   # Treine o LightGBM com 100 trials de otimização para máxima precisão
   python train.py --n_trials 100
   ```
   Ao final, o arquivo `sales_forecaster_v2_final.joblib` será criado na pasta `artifacts/`.

**2. Gerar o Arquivo de Submissão Final:**
   *Este script carrega o modelo já treinado e gera o arquivo final feito para a submissão no Hackathon (limitado a 1,5 milhão de linhas).*

   * **Para gerar o arquivo de SUBMISSÃO (limitado a 1.5M de linhas):**
       ```bash
       python predict.py
       ```
       Este é o comando padrão e gerará o arquivo formatado para a plataforma do hackathon.

   * **Para gerar a previsão COMPLETA (Opcional):**
       Se desejar a previsão para todos os produtos, sem o limite de linhas, use a flag `--full_forecast`.
       ```bash
       python predict.py --full_forecast
       ```

---

## 📊 Resultados

O modelo final, avaliado em um conjunto de validação hold-out (últimas 4 semanas de 2022), alcançou um **Erro Médio Absoluto (MAE)** de **2.576895**. Este valor indica que, em média, as previsões do modelo erraram por aproximadamente 2,5 unidades, uma métrica de alta precisão para a complexidade do problema.

---

## 💻 Tecnologias Utilizadas

* **Linguagem:** Python 3.13
* **Bibliotecas Principais:**
    * Pandas (Manipulação de Dados)
    * **LightGBM** (Modelagem de Gradient Boosting)
    * **Optuna** (Otimização de Hiperparâmetros)
    * Scikit-learn (Métricas e Pré-processamento)
    * Joblib (Serialização de Modelos)
    * NumPy (Computação Numérica)

---

## ✍️ Autores - Equipe: BSB Data 01

* **Erick Cardoso Mendes (desenvolvedor)**
* **Pedro Morato Lahoz (relator)**

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT.
