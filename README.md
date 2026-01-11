# Repositório de Experimentos de Machine Learning & MLOps

Este repositório é dedicado a experimentos de Machine Learning e já possui uma estrutura de MLOps completa e pronta para uso. Basta configurar suas credenciais e começar a rodar.

O projeto integra ferramentas de ponta como **FLAML**, **AutoGluon**, **Auto-sklearn**, **H2O AutoML** e **TPOT** com rastreamento automático via **MLflow**, **DagsHub** e **Weights & Biases**.

## 🎯 MLOps Enterprise Framework (V4.0)

Este projeto fornece uma integração completa de pipeline de treinamento com as seguintes capacidades:

- **End-to-end training pipeline**: Do carregamento de dados ao registro do modelo.
- **Unified AutoML Engine**: Suporte nativo para **AutoGluon**, **FLAML**, **TPOT**, **Auto-sklearn** e **H2O AutoML**.
- **Data Validation & Integrity**: Verificação de tipos, nulos e integridade antes do treinamento.
- **Data Drift Detection**: Monitoramento contínuo de drift de dados usando **Evidently AI**.
- **Experiment Tracking**: Rastreamento detalhado de métricas, parâmetros e artefatos no **MLflow**, **DagsHub** e **W&B**.
- **Explainability (XAI)**: Transparência total com **SHAP** e **LIME**.
- **Inference Optimization**: Exportação automática para o formato **ONNX**.
- **Automated Serving**: Geração de scripts de API pronta para produção com **Flask** e **ONNX Runtime**.

## 🚀 Funcionalidades

- **ML Clássico**: Suporte total a tarefas de Classificação e Regressão.
- **Visão Computacional (CV)**: Integração com **YOLOv8** para detecção de objetos.
- **Processamento de Linguagem Natural (NLP)**: Suporte a **Transformers** e datasets do **HuggingFace**.
- **Séries Temporais**: Suporte inicial via **Prophet** e motores de AutoML.
- **Testes Automatizados**: Suíte completa de testes unitários, de integração e aceitação.

## 🛠️ Tecnologias

- **Frameworks**: Scikit-learn, PyTorch, Transformers, Prophet, Ultralytics (YOLO)
- **MLOps**: MLflow, DagsHub, Weights & Biases, Optuna, Evidently AI
- **Serving & Export**: Flask, ONNX, ONNX Runtime, skl2onnx, Gradio
- **Inference**: SHAP, LIME

## 📂 Estrutura do Projeto

- `train_and_save_professional.py`: Framework universal (Core V4.0).
- `experiments/senti-pred/senti_comparison.py`: Benchmark de AutoML para análise de sentimento.
- `tests/`: Suíte de testes automatizados (`pytest`).
- `.env.example`: Modelo para configuração de variáveis de ambiente.
- `requirements.txt`: Dependências completas e atualizadas.

## 🛠️ Instalação e Uso

1. **Crie um ambiente virtual e instale as dependências**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # ou venv\Scripts\activate no Windows
   pip install -r requirements.txt
   ```

2. **Configure suas credenciais**:
   Copie o arquivo `.env.example` para `.env` e preencha suas chaves (DagsHub, MLflow, W&B).

3. **Execute um treinamento**:
   ```python
   from train_and_save_professional import MLOpsEnterprise
   import pandas as pd

   mlops = MLOpsEnterprise()
   # Treinar usando H2O AutoML
   model, score = mlops.train_automl("seu_dataset.csv", engine="h2o", timeout=300)
   ```

4. **Interface Visual (Dashboard)**:
   ```bash
   python gradio_app.py
   ```
   Acesse em seu navegador: `http://localhost:7860`

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
