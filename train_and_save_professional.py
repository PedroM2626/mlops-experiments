#!/usr/bin/env python3
"""
🎯 MLOPS PROFESSIONAL - SCRIPT ÚNICO
Treina, salva e versiona modelos com MLflow-DagsHub
Script profissional que substitui múltiplos scripts
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import dagshub
from dotenv import load_dotenv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLOpsProfessional:
    """
    Classe profissional para MLOps com MLflow-DagsHub
    """
    
    def __init__(self, repo_owner='PedroM2626', repo_name='experiments'):
        """Inicializa MLflow com DagsHub"""
        load_dotenv()
        
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.dagshub_token = os.getenv('DAGSHUB_TOKEN', '')
        
        # Configurar DagsHub
        self._setup_dagshub()
        
        print(f"✅ MLOps Professional inicializado")
        print(f"🔗 DagsHub: https://dagshub.com/{repo_owner}/{repo_name}")
    
    def _setup_dagshub(self):
        """Configura MLflow com DagsHub"""
        try:
            # Inicializar DagsHub (isso configura tracking URI e artefatos se mlflow=True)
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
            
            # Garantir URI de tracking correto
            tracking_uri = f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            
            # Configurar credenciais para o storage de artefatos (S3 compatível do DagsHub)
            if self.dagshub_token:
                os.environ['MLFLOW_TRACKING_USERNAME'] = self.repo_owner
                os.environ['MLFLOW_TRACKING_PASSWORD'] = self.dagshub_token
                
                # O DagsHub usa o token como senha para o storage de artefatos também
                os.environ['AWS_ACCESS_KEY_ID'] = self.repo_owner
                os.environ['AWS_SECRET_ACCESS_KEY'] = self.dagshub_token
                
                # IMPORTANTE: Definir o endpoint S3 do DagsHub para o MLflow
                os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.s3"
                
                dagshub.auth.add_app_token(self.dagshub_token)
                
            print(f"✅ MLflow e Artefatos configurados com DagsHub")
            
        except Exception as e:
            print(f"⚠️  Erro ao configurar DagsHub, usando local: {e}")
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    def train_model(self, model_type='rf', experiment_name='sentiment_analysis'):
        """
        Treina um modelo e salva com versionamento
        
        Args:
            model_type: 'rf', 'logistic', 'knn', 'linearsvc'
            experiment_name: nome do experimento
        """
        print(f"\n🚀 Treinando modelo: {model_type}")
        print("="*60)
        
        # Carregar dados
        print("📁 Carregando dados...")
        df = pd.read_csv('processed_train.csv')
        print(f"✅ {len(df)} amostras de treino")
        
        # Preparar dados
        X = df['text_lemmatized'].fillna('')
        y = df['sentiment']
        
        # Dividir dados
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Configurar experimento
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"📊 Run ID: {run_id}")
            
            # Criar modelo baseado no tipo
            if model_type == 'logistic':
                model = LogisticRegression(random_state=42, max_iter=1000)
                model_name = "sentiment_logistic"
            elif model_type == 'rf':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model_name = "sentiment_rf"
            elif model_type == 'knn':
                model = KNeighborsClassifier(n_neighbors=5)
                model_name = "sentiment_knn"
            elif model_type == 'linearsvc':
                model = LinearSVC(random_state=42)
                model_name = "sentiment_linearsvc"
            else:
                raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
            
            # Criar pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', model)
            ])
            
            # Treinar modelo
            print(f"🎯 Treinando {model_type}...")
            pipeline.fit(X_train, y_train)
            
            # Avaliar
            y_pred = pipeline.predict(X_val)
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0)
            }
            
            # Logar parâmetros
            params = {
                'model_type': model_type,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'features': 'text_tfidf',
                'max_features': 5000,
                'random_state': 42
            }
            
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Logar métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                print(f"  📈 {metric_name}: {metric_value:.4f}")
            
            # Gerar e logar artefatos extras
            print("📝 Gerando artefatos extras...")
            import shutil
            temp_dir = "temp_artifacts"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # 1. Relatório de Classificação
                report = classification_report(y_val, y_pred, output_dict=True)
                report_path = f"{temp_dir}/classification_report.json"
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact(report_path, "metrics")
                
                # 2. Matriz de Confusão
                cm = confusion_matrix(y_val, y_pred)
                cm_path = f"{temp_dir}/confusion_matrix.json"
                with open(cm_path, "w") as f:
                    json.dump(cm.tolist(), f, indent=4)
                mlflow.log_artifact(cm_path, "metrics")
                
                # Tags
                tags = {
                    'task': 'sentiment_analysis',
                    'model_name': model_name,
                    'saved_at': datetime.now().isoformat(),
                    'framework': 'sklearn',
                    'experiment_type': 'mlops_professional'
                }
                
                for tag_name, tag_value in tags.items():
                    mlflow.set_tag(tag_name, tag_value)
                
                # Registrar modelo com versionamento
                print(f"📦 Gerando pacote do modelo (MLmodel, conda.yaml, etc)...")
                model_package_dir = os.path.join(temp_dir, "model_package")
                if os.path.exists(model_package_dir):
                    shutil.rmtree(model_package_dir)
                
                # Salva o modelo localmente com todos os arquivos de configuração
                mlflow.sklearn.save_model(
                    sk_model=pipeline,
                    path=model_package_dir
                )
                
                # Loga a pasta inteira como artefatos
                print(f"🚀 Fazendo upload do pacote completo para o DagsHub...")
                mlflow.log_artifacts(model_package_dir, artifact_path="model")
                
                # Também registra no Model Registry para versionamento
                mlflow.sklearn.log_model(
                    sk_model=pipeline, 
                    artifact_path="model_registry",
                    registered_model_name=model_name
                )
                
                # Salvar cópia local e também logar como artefato direto para download fácil
                os.makedirs("models/professional", exist_ok=True)
                local_path = f"models/professional/{model_name}_{run_id[:8]}.pkl"
                joblib.dump(pipeline, local_path)
                mlflow.log_artifact(local_path, "direct_download")
                print(f"  💾 Salvo localmente: {local_path}")
                
                print(f"✅ Modelo {model_name} salvo com sucesso!")
            
            finally:
                 # Limpeza de arquivos temporários locais
                 if os.path.exists(temp_dir):
                     shutil.rmtree(temp_dir)
                     print(f"  🧹 Pasta {temp_dir} removida.")
            
            return {
                'model_name': model_name,
                'run_id': run_id,
                'metrics': metrics,
                'local_path': local_path
            }
    
    def save_existing_model(self, model_path, model_name, experiment_name='existing_models'):
        """Salva um modelo existente com versionamento"""
        
        print(f"\n💾 Salvando modelo existente: {model_name}")
        print("="*60)
        
        # Carregar modelo
        print(f"📦 Carregando modelo de: {model_path}")
        model = joblib.load(model_path)
        
        # Carregar dados de validação
        print("📁 Carregando dados de validação...")
        df_val = pd.read_csv('processed_validation.csv')
        print(f"✅ {len(df_val)} amostras de validação")
        
        # Preparar dados baseado no tipo de modelo
        if 'knn' in model_path.lower():
            X_val = df_val.drop(['sentiment'], axis=1)
            feature_type = 'numeric + categorical'
        else:
            X_val = df_val['text_lemmatized'].fillna('')
            feature_type = 'text features'
        
        y_val = df_val['sentiment']
        
        # Avaliar modelo
        print("📊 Avaliando modelo...")
        y_pred = model.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0)
        }
        
        # Configurar experimento
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"📊 Run ID: {run_id}")
            
            # Logar parâmetros
            params = {
                'model_source': 'existing_model',
                'features': feature_type,
                'validation_samples': len(y_val),
                'classes': len(y_val.unique()),
                'original_file': os.path.basename(model_path)
            }
            
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Logar métricas
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                print(f"  📈 {metric_name}: {metric_value:.4f}")
            
            # Gerar e logar artefatos extras
            print("📝 Gerando artefatos extras...")
            temp_dir = "temp_artifacts"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # 1. Relatório de Classificação
                report = classification_report(y_val, y_pred, output_dict=True)
                report_path = os.path.join(temp_dir, "classification_report_existing.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact(report_path, "metrics")
                
                # 2. Matriz de Confusão
                cm = confusion_matrix(y_val, y_pred)
                cm_path = os.path.join(temp_dir, "confusion_matrix_existing.json")
                with open(cm_path, "w") as f:
                    json.dump(cm.tolist(), f, indent=4)
                mlflow.log_artifact(cm_path, "metrics")
                
                # Tags
                tags = {
                    'model_name': model_name,
                    'original_file': os.path.basename(model_path),
                    'saved_at': datetime.now().isoformat(),
                    'task': 'sentiment_analysis',
                    'model_source': 'existing_model'
                }
                
                for tag_name, tag_value in tags.items():
                    mlflow.set_tag(tag_name, tag_value)
                
                # Registrar modelo com versionamento
                print(f"📦 Gerando pacote do modelo (MLmodel, conda.yaml, etc)...")
                model_package_dir = os.path.join(temp_dir, "model_package")
                if os.path.exists(model_package_dir):
                    shutil.rmtree(model_package_dir)
                
                # Salva o modelo localmente com todos os arquivos de configuração
                mlflow.sklearn.save_model(
                    sk_model=model,
                    path=model_package_dir
                )
                
                # Loga a pasta inteira como artefatos
                print(f"🚀 Fazendo upload do pacote completo para o DagsHub...")
                mlflow.log_artifacts(model_package_dir, artifact_path="model")
                
                # Também registra no Model Registry para versionamento
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path="model_registry",
                    registered_model_name=model_name
                )
                
                # Salvar cópia local e também logar como artefato direto para download fácil
                os.makedirs("models/professional", exist_ok=True)
                local_path = f"models/professional/{model_name}_{run_id[:8]}.pkl"
                joblib.dump(model, local_path)
                mlflow.log_artifact(local_path, "direct_download")
                print(f"  💾 Salvo localmente: {local_path}")
                
            finally:
                 # Limpeza de arquivos temporários locais
                 if os.path.exists(temp_dir):
                     shutil.rmtree(temp_dir)
                     print(f"  🧹 Pasta {temp_dir} removida.")
            
            return {
                'model_name': model_name,
                'run_id': run_id,
                'metrics': metrics,
                'local_path': local_path
            }
    
    def compare_models(self, model_names):
        """Compara modelos salvos no DagsHub"""
        
        print(f"\n📊 Comparando modelos: {model_names}")
        print("="*60)
        
        client = mlflow.tracking.MlflowClient()
        
        comparison_data = []
        
        for model_name in model_names:
            try:
                # Obter versão mais recente
                latest_versions = client.get_latest_versions(model_name)
                if not latest_versions:
                    print(f"⚠️  Modelo {model_name} não encontrado")
                    continue
                
                model_version = latest_versions[0]
                run = client.get_run(model_version.run_id)
                
                comparison_data.append({
                    'model_name': model_name,
                    'version': model_version.version,
                    'run_id': model_version.run_id[:8],
                    'metrics': dict(run.data.metrics),
                    'params': dict(run.data.params)
                })
                
            except Exception as e:
                print(f"❌ Erro ao processar {model_name}: {e}")
        
        if comparison_data:
            print("\n📊 Tabela de Comparação:")
            print(f"{'Modelo':<35} {'Versão':<8} {'Accuracy':<10} {'F1-Score':<10} {'Tipo'}")
            print("-" * 80)
            
            for data in comparison_data:
                metrics = data['metrics']
                accuracy = metrics.get('accuracy', 0)
                f1_score = metrics.get('f1_score', 0)
                model_type = data['params'].get('model_type', 'Unknown')
                print(f"{data['model_name']:<35} {data['version']:<8} {accuracy:<10.4f} {f1_score:<10.4f} {model_type}")
            
            # Encontrar melhor modelo
            best_model = max(comparison_data, key=lambda x: x['metrics'].get('accuracy', 0))
            print(f"\n🏆 Melhor modelo: {best_model['model_name']}")
            print(f"   Accuracy: {best_model['metrics']['accuracy']:.4f}")
            print(f"   F1-Score: {best_model['metrics']['f1_score']:.4f}")
            
            return best_model
        
        return None
    
    def list_all_models(self):
        """Lista todos os modelos no Model Registry"""
        
        print(f"\n🏛️  Modelos no Registry")
        print("="*60)
        
        client = mlflow.tracking.MlflowClient()
        
        try:
            registered_models = client.search_registered_models()
            
            print(f"📦 Total de modelos: {len(registered_models)}")
            
            for i, model in enumerate(registered_models, 1):
                print(f"\n{i}. {model.name}")
                
                if model.latest_versions:
                    for version in model.latest_versions:
                        print(f"   v{version.version} - {version.status}")
                        print(f"      Run ID: {version.run_id[:8]}...")
                        print(f"      Criado: {version.creation_timestamp}")
        
        except Exception as e:
            print(f"❌ Erro ao listar modelos: {e}")

def main():
    """Função principal demonstrativa"""
    
    print("🎯 MLOPS PROFESSIONAL - SCRIPT ÚNICO")
    print("="*70)
    
    # Inicializar MLOps
    mlops = MLOpsProfessional()
    
    # Treinar novos modelos
    models_to_train = ['logistic', 'rf', 'knn', 'linearsvc']
    trained_models = []
    
    for model_type in models_to_train:
        try:
            result = mlops.train_model(model_type)
            trained_models.append(result['model_name'])
        except Exception as e:
            print(f"❌ Erro ao treinar {model_type}: {e}")
    
    # Salvar modelos existentes
    existing_models = [
        ('models/active/senti-pred-free-mlops-knn.pkl', 'sentiment_knn_existing'),
        ('models/active/senti-pred-linearsvc.pkl', 'sentiment_linearsvc_existing')
    ]
    
    for model_path, model_name in existing_models:
        if os.path.exists(model_path):
            try:
                result = mlops.save_existing_model(model_path, model_name)
                trained_models.append(result['model_name'])
            except Exception as e:
                print(f"❌ Erro ao salvar {model_name}: {e}")
    
    # Comparar todos os modelos
    if trained_models:
        print(f"\n{'='*70}")
        print("🏆 COMPARAÇÃO FINAL DE MODELOS")
        print("="*70)
        
        best_model = mlops.compare_models(trained_models)
        
        if best_model:
            print(f"\n✅ Melhor modelo identificado: {best_model['model_name']}")
            print(f"🔗 Acesse: https://dagshub.com/PedroM2626/experiments.mlflow")
    
    # Listar todos os modelos
    mlops.list_all_models()
    
    print(f"\n{'='*70}")
    print("✅ PROCESSO CONCLUÍDO")
    print("="*70)
    
    print(f"\n🌐 Acesse seus experimentos:")
    print(f"   DagsHub: https://dagshub.com/PedroM2626/experiments")
    print(f"   MLflow: https://dagshub.com/PedroM2626/experiments.mlflow")
    
    print(f"\n💡 Comandos úteis:")
    print(f"   • Visualizar interface: python -m mlflow ui --port 5000")
    print(f"   • Treinar modelo específico: python train_and_save_professional.py --model rf")
    print(f"   • Salvar modelo existente: python train_and_save_professional.py --existing models/meu_modelo.pkl --name meu_modelo")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MLOps Professional - Script Único')
    parser.add_argument('--model', type=str, choices=['rf', 'logistic', 'knn', 'linearsvc'], 
                       help='Tipo de modelo para treinar')
    parser.add_argument('--existing', type=str, help='Caminho para modelo existente')
    parser.add_argument('--name', type=str, help='Nome para o modelo existente')
    parser.add_argument('--compare', nargs='+', help='Comparar modelos específicos')
    parser.add_argument('--list', action='store_true', help='Listar todos os modelos')
    
    args = parser.parse_args()
    
    mlops = MLOpsProfessional()
    
    if args.model:
        # Treinar modelo específico
        result = mlops.train_model(args.model)
        print(f"✅ Modelo {args.model} treinado com sucesso!")
    
    elif args.existing and args.name:
        # Salvar modelo existente
        result = mlops.save_existing_model(args.existing, args.name)
        print(f"✅ Modelo {args.name} salvo com sucesso!")
    
    elif args.compare:
        # Comparar modelos específicos
        best = mlops.compare_models(args.compare)
        if best:
            print(f"🏆 Melhor modelo: {best['model_name']}")
    
    elif args.list:
        # Listar modelos
        mlops.list_all_models()
    
    else:
        # Executar demonstração completa
        main()