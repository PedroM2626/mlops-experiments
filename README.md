# Repositório de Experimentos de Machine Learning & MLOps

Este repositório é dedicado a experimentos de Machine Learning e já possui uma estrutura de MLOps completa e pronta para uso. Basta configurar suas credenciais e começar a rodar.

O projeto integra ferramentas de ponta como **FLAML** e **AutoGluon** com rastreamento automático via **MLflow** e **DagsHub**.

## 🌟 Training Pipeline Integration & Features

Este projeto fornece uma integração completa de pipeline de treinamento com as seguintes capacidades:

- **End-to-end training pipeline**: Do carregamento de dados ao registro do modelo.
- **Automated model training and registration**: Uso de AutoML para encontrar os melhores modelos e salvá-los automaticamente.
- **Training job orchestration**: Scripts modulares para orquestrar múltiplas ferramentas.
- **Experiment tracking integration**: Rastreamento detalhado de métricas, parâmetros e artefatos no MLflow/DagsHub.

## 🎯 Supported Use Cases

A arquitetura e as bibliotecas integradas (FLAML e AutoGluon) suportam uma ampla gama de tarefas. 
*Atualmente, os scripts de exemplo implementam **Classificação** e **Regressão**, mas podem ser facilmente adaptados para:*

### Classification
- Customer churn prediction
- Fraud detection
- Sentiment analysis
- Image classification
- Spam detection

### Regression
- Demand forecasting
- Price prediction
- Sales forecasting
- Resource usage prediction

### Time Series (Suportado pelas libs)
- Stock/crypto prediction
- Sensor data forecasting
- Energy consumption
- Traffic prediction

### Other
- Recommendation systems
- Anomaly detection

## 🚀 Funcionalidades

- **MLOps Universal**: Um único script (`train_and_save_professional.py`) para NLP Clássico, Transformers e Visão Computacional.
- **SOTA NLP**: Suporte nativo a Transformers (Hugging Face / BERT).
- **Deep Learning & CV**: Integração com PyTorch para classificação de imagens (ResNet/MobileNet).
- **Explainability (XAI)**: Explicações de modelos via **SHAP**.
- **Inference Ready**: Exportação para o formato universal **ONNX**.
- **Model Registry**: Gestão completa de versões no DagsHub/MLflow.
- **Artefatos Visuais**: Geração automática de Matrizes de Confusão e Curvas de Loss.

---

## 🛠️ Tecnologias

- **Frameworks**: Scikit-learn, PyTorch, Transformers (Hugging Face)
- **MLOps**: MLflow, DagsHub, ONNX
- **Explainability**: SHAP
- **Data**: Pandas, Numpy, Datasets (HF)
- **Visualização**: Matplotlib, Seaborn

---

## 📂 Estrutura do Projeto

- `train_and_save_professional.py`: Framework universal de treinamento e versionamento.
- `compare_essential.py`: Comparação de métricas entre modelos registrados.
- `.env`: Configurações de acesso e tokens.

## 🛠️ Instalação

1. **Clone o repositório** (se aplicável) ou navegue até a pasta do projeto.

2. **Crie um ambiente virtual** (recomendado):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```
   *Nota: O AutoGluon pode exigir dependências específicas do sistema operacional. Consulte a [documentação oficial](https://auto.gluon.ai/stable/index.html) se encontrar problemas.*

4. **Configuração do Ambiente**:
   Crie um arquivo `.env` baseado no `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Preencha com suas credenciais do DagsHub:
   - `MLFLOW_TRACKING_URI`: URL do seu repositório no DagsHub (aba "Remote" -> "MLflow").
   - `MLFLOW_TRACKING_USERNAME`: Seu usuário DagsHub.
   - `MLFLOW_TRACKING_PASSWORD`: Seu token ou senha DagsHub.

   *Se você já rodou `dagshub login` na máquina, o script tentará usar a configuração global, mas o arquivo `.env` é recomendado para reprodutibilidade.*

## 🎯 Como Usar

### 1. NLP Clássico (Sklearn)
```bash
python train_and_save_professional.py --task nlp
```

### 2. Transformers (BERT/DistilBERT)
```bash
python train_and_save_professional.py --task transformer
```

### 3. Visão Computacional (ResNet/PyTorch)
```bash
python train_and_save_professional.py --task cv
```

### 4. Executar Todo o Pipeline
```bash
python train_and_save_professional.py --task all
```

## 📊 Resultados

Após a execução, você pode visualizar os resultados:
1. No terminal, será exibida a performance dos modelos.
2. Acesse a interface do MLflow no seu repositório DagsHub para ver gráficos, métricas comparativas e baixar os modelos treinados.

## 📚 Documentação das Ferramentas

Aqui estão os links para a documentação oficial de cada ferramenta utilizada neste projeto:

### FLAML (Microsoft)
- [Documentação Oficial](https://microsoft.github.io/FLAML/docs/Getting-Started)
- [Repositório GitHub](https://github.com/microsoft/FLAML)
- [Guia de Integração com MLflow](https://microsoft.github.io/FLAML/docs/reference/fabric/mlflow/)

### AutoGluon (Amazon)
- [Documentação Oficial](https://auto.gluon.ai/stable/index.html)
- [Repositório GitHub](https://github.com/autogluon/autogluon)
- [Guia de Integração com MLflow](https://community.databricks.com/t5/machine-learning/autogluon-mlflow-integration/td-p/111423)

### MLflow
- [Documentação Oficial](https://mlflow.org/docs/latest/)
- [Guia de Autologging](https://mlflow.org/docs/latest/ml/tracking/autolog/)
- [Integração com DagsHub](https://dagshub.com/docs/mlflow/)

### DagsHub
- [Documentação Oficial](https://dagshub.com/docs/)
- [Guia de Configuração do MLflow](https://dagshub.com/docs/mlflow/)

## 🧪 Testes

Para garantir que tudo está funcionando corretamente, execute os testes unitários:

```bash
pytest
```

## 🏛️ Sistema de Versionamento de Modelos

O projeto agora inclui um sistema completo de versionamento de modelos via MLflow, permitindo:

### ✅ Funcionalidades de Versionamento

- **Salvamento automático com versionamento**: Cada experimento salva uma nova versão do modelo
- **Model Registry profissional**: Registro centralizado de todos os modelos e versões
- **Métricas por versão**: Cada versão inclui métricas completas de avaliação
- **Carregamento por versão**: Possibilidade de carregar modelos específicos por nome e versão
- **Backup local**: Cópias locais de todos os modelos salvos

### 📋 Modelos Salvos

Atualmente, o sistema possui os seguintes modelos com versionamento:

| Modelo | Versão | Accuracy | F1-Score | Tipo |
|--------|--------|----------|----------|------|
| `sentiment_knn_original` | v1 | 0.9990 | 0.9990 | K-Nearest Neighbors |
| `sentiment_linearsvc_original` | v1 | 0.9380 | 0.9380 | Linear SVC |
| `sentiment_logistic_v20260106_1340` | v1 | 0.6759 | 0.6732 | Logistic Regression |
| `sentiment_rf_v20260106_1341` | v1 | 0.8501 | 0.8501 | Random Forest |

### 🚀 Como Usar o MLOps Professional (Script Único)

O projeto agora conta com um script único profissional que gerencia todo o ciclo de vida:

```bash
# 1. Listar todos os modelos no registry
python train_and_save_professional.py --list

# 2. Treinar novo modelo (rf, logistic, knn, linearsvc)
python train_and_save_professional.py --model rf

# 3. Salvar modelo existente no MLflow
python train_and_save_professional.py --existing models/meu_modelo.pkl --name meu_modelo

# 4. Comparar modelos de um experimento
python train_and_save_professional.py --compare sentiment_analysis

# 5. Executar demonstração completa
python train_and_save_professional.py
```

### 🧹 Gestão de Arquivos e Limpeza Automática
O script profissional foi projetado para manter seu ambiente local limpo:
- **`temp_artifacts/`**: Criada apenas durante a execução para gerar JSONs de métricas e removida imediatamente após o upload para o DagsHub.
- **`mlruns/`**: Cache local do MLflow é limpo automaticamente após a sincronização bem-sucedida com o servidor remoto.
- **`models/professional/`**: Único diretório que mantém cópias locais dos modelos treinados para uso offline imediato.

### 📦 Gerenciamento de Artefatos (DagsHub)

As métricas, parâmetros e **artefatos** (modelos, matriz de confusão, relatórios) são enviados automaticamente para o DagsHub.

**Configuração necessária:**
Certifique-se de que seu arquivo `.env` contenha o `DAGSHUB_TOKEN`:
```env
DAGSHUB_TOKEN=seu_token_aqui
```
O script configura automaticamente o storage S3 compatível do DagsHub para garantir que os arquivos `.pkl` e `.json` subam corretamente.

#### 5. Visualizar interface web do MLflow
```bash
python -m mlflow ui --port 5000
# Acesse: http://localhost:5000
```

### 🔍 Carregando Modelos por Versão

```python
import mlflow.sklearn

# Carregar última versão
model = mlflow.sklearn.load_model("models:/sentiment_knn_original/latest")

# Carregar versão específica
model = mlflow.sklearn.load_model("models:/sentiment_knn_original/1")
```

### 📊 Comparação de Modelos

O sistema permite comparação automática entre todas as versões salvas, mostrando:
- Métricas de avaliação (accuracy, precision, recall, f1-score)
- Parâmetros utilizados
- Data de criação
- Status do modelo

### 💡 Próximos Passos

1. **Executar experimentos**: Cada novo treinamento gera uma versão automaticamente
2. **Comparar versões**: Use os scripts de comparação para escolher o melhor modelo
3. **Fazer deploy**: Carregue a versão desejada para produção
4. **Monitorar**: Acompanhe a performance de cada versão

O sistema está totalmente integrado com o pipeline MLOps existente, garantindo que cada experimento (treinamento, fine-tuning, etc.) seja devidamente versionado e rastreado.
