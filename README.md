---
title: MLOps Enterprise Dashboard
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.12.0
app_file: app.py
pinned: false
---

# Repositório de Experimentos de Machine Learning & MLOps

Este repositório é dedicado a experimentos de Machine Learning e mlops. já possui uma estrutura de MLOps completa e pronta para uso. Basta configurar suas credenciais e começar a rodar.

O projeto integra ferramentas de ponta como **FLAML**, **AutoGluon**, **Auto-sklearn**, **H2O AutoML**, **TPOT** e **ZenML** com rastreamento automático via **MLflow**, **DagsHub** e **Weights & Biases**.

## 🎯 MLOps Enterprise Framework (V7.0)

Este projeto fornece uma integração completa de pipeline de treinamento com as seguintes capacidades:

- **End-to-end training pipeline**: Do carregamento de dados ao registro do modelo, agora com orquestração **ZenML**.
- **Unified AutoML Engine**: Suporte nativo para **AutoGluon**, **FLAML**, **TPOT**, **Auto-sklearn**, **H2O AutoML** e modo **Unified**.
- **Manual Training & Optuna**: Treinamento manual com escolha de modelos (RF, XGBoost) e otimização de hiperparâmetros.
- **Deep Learning Tabular**: Redes neurais em PyTorch para dados tabulares.
- **Data Validation & Integrity**: Verificação de tipos, nulos e integridade (NaN/Inf) antes do treinamento.
- **Data Drift Detection**: Monitoramento contínuo de drift de dados usando **Evidently AI**.
- **Experiment Tracking**: Rastreamento detalhado de métricas, parâmetros e artefatos no **MLflow**, **DagsHub** e **W&B**.
- **Explainability (XAI)**: Transparência total com **SHAP** e **LIME**.
- **Inference Optimization**: Exportação automática para o formato **ONNX**.
- **Automated Serving**: Geração de scripts de API pronta para produção com **Flask** e **ONNX Runtime**.

## 🚀 Funcionalidades

- **ML Clássico**: Suporte total a tarefas de Classificação e Regressão.
- **Visão Computacional (CV)**: Integração com **YOLOv8** para detecção de objetos (Geral e Facial) e detecção em tempo real (Webcam).
- **Processamento de Linguagem Natural (NLP)**: Suporte a **Transformers** e datasets do **HuggingFace**.
- **Séries Temporais**: Suporte inicial via **Prophet** e motores de AutoML.
- **Orquestração**: Pipelines reprodutíveis com **ZenML**.
- **Testes Automatizados**: Suíte completa de testes unitários, de integração e aceitação.

## 🛠️ Tecnologias

- **Frameworks**: Scikit-learn, PyTorch, Transformers, Prophet, Ultralytics (YOLO)
- **MLOps**: MLflow, DagsHub, Weights & Biases, Optuna, Evidently AI, ZenML
- **Serving & Export**: Flask, ONNX, ONNX Runtime, skl2onnx, Gradio
- **Inference**: SHAP, LIME

## 📂 Estrutura do Projeto

- `train_and_save_professional.py`: Framework universal (Core V7.0).
- `app.py`: Ponto de entrada para o Hugging Face Spaces (Gradio).
- `gradio_app.py`: Interface principal do Dashboard.
- `requirements.txt`: Dependências completas e atualizadas.

### ⚠️ Notas de Compatibilidade
- **NumPy 2.x**: O framework foi otimizado para **NumPy < 2.0.0**.
- **auto-sklearn**: Esta biblioteca foi removida do `requirements.txt` padrão pois não é compatível com **Windows** e causa erros de build no **Hugging Face Spaces**. Se você estiver no **Linux** e desejar usá-la, instale manualmente: `pip install auto-sklearn`.

🎯 MLOPS ENTERPRISE - UNIVERSAL FRAMEWORK (V7.0)
==============================================

## 🚀 Recursos do Framework Universal (V7.0)

### 🤖 AutoML Unificado
- **Engines Suportadas**: FLAML, AutoGluon, TPOT, Auto-sklearn e H2O AutoML.
- **Modos**: Classificação e Regressão com exportação automática para **ONNX**.

### 🛤️ Orquestração com ZenML
- **Pipelines**: Criação de fluxos de trabalho de ML reprodutíveis e modulares.

### 📈 Séries Temporais (Time Series)
- **Engine**: Integração com **Prophet** para previsões de demanda e tendências.

### 💎 Aprendizado Não Supervisionado
- **Clustering**: Agrupamento inteligente com **K-Means** e **DBSCAN**.
- **Detecção de Anomalias**: Identificação de outliers com **Isolation Forest**.

### 🧠 Fine-Tuning Avançado
- **NLP**: Ajuste fino (fine-tuning) de modelos **Transformers**.

### 📸 Visão Computacional (CV)
- **Detecção Facial**: YOLOv8 para identificação em tempo real.

### 📝 Processamento de Linguagem Natural (NLP)
- Pipelines prontos para Sentimento, Sumarização e NER.

### 📈 MLOps & Dashboard
- **Integrações**: MLflow, DagsHub, W&B.
- **Dashboard**: Interface Gradio completa (V7.0).

## 🛠️ Instalação e Uso

1. **Crie um ambiente virtual e instale as dependências**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # ou venv\Scripts\activate no Windows
   pip install -r requirements.txt
   ```

2. **Configure suas credenciais**:
   Copie o arquivo `.env.example` para `.env` e preencha suas chaves (DagsHub, MLflow, W&B).

3. **Execute o Dashboard**:
   ```bash
   python app.py
   ```

5. **Rodar o Benchmark de Sentimento**:
   ```bash
   python experiments/senti-pred/senti_comparison.py
   ```

5. **Executar Testes**:
   ```bash
   pytest tests/
   ```

## 🧪 Testes Automatizados

O projeto inclui uma suíte de testes robusta que cobre:
- **Unitários**: Validação de inicialização e lógica de dados.
- **Integração**: Conexão com MLflow e execução de motores de AutoML.
- **Aceitação**: Fluxo completo de validação, treino, explicação e exportação.

---
Desenvolvido para máxima produtividade em projetos de Ciência de Dados e MLOps.