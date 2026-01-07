import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import dagshub
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
import shutil
from dotenv import load_dotenv

class DLCVProfessional:
    def __init__(self):
        load_dotenv()
        self.repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
        self.repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")
        self.dagshub_token = os.getenv("DAGSHUB_TOKEN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_dagshub()

    def _setup_dagshub(self):
        """Configura MLflow e S3 com DagsHub para Deep Learning"""
        try:
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
            
            # Credenciais para S3 Artefatos
            if self.dagshub_token:
                os.environ['AWS_ACCESS_KEY_ID'] = self.repo_owner
                os.environ['AWS_SECRET_ACCESS_KEY'] = self.dagshub_token
                os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.s3"
                dagshub.auth.add_app_token(self.dagshub_token)
            
            print(f"✅ Deep Learning MLOps configurado no DagsHub ({self.device})")
        except Exception as e:
            print(f"⚠️ Erro ao configurar DagsHub: {e}")
            mlflow.set_tracking_uri("sqlite:///mlflow_dl.db")

    def get_transfer_learning_model(self, model_name='resnet18', num_classes=2):
        """Prepara um modelo pré-treinado para Transfer Learning"""
        print(f"📦 Carregando modelo base: {model_name}")
        
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'mobilenet':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        
        return model.to(self.device)

    def train_model(self, model, train_loader, val_loader, criterion, optimizer, num_epochs=5, experiment_name="CV_Experiments"):
        """Treina o modelo com rastreamento completo no MLflow"""
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"📊 Run ID: {run_id}")
            
            # Logar Hiperparâmetros
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("device", str(self.device))
            mlflow.log_param("optimizer", type(optimizer).__name__)
            mlflow.log_param("criterion", type(criterion).__name__)
            
            best_acc = 0.0
            history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
            
            for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}")
                
                # Fase de Treino
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                
                epoch_loss = running_loss / len(train_loader.dataset)
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                history['train_loss'].append(epoch_loss)
                
                # Fase de Validação
                model.eval()
                val_loss = 0.0
                corrects = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        corrects += torch.sum(preds == labels.data)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                epoch_val_loss = val_loss / len(val_loader.dataset)
                epoch_acc = corrects.double() / len(val_loader.dataset)
                
                mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
                mlflow.log_metric("val_acc", float(epoch_acc), step=epoch)
                
                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(float(epoch_acc))
                
                print(f"  Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_acc:.4f}")
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
            
            # --- Gerar Artefatos ---
            print("📝 Gerando artefatos de Visão Computacional...")
            temp_dir = "temp_dl_artifacts"
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            # 1. Curva de Aprendizado
            plt.figure(figsize=(10, 5))
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Learning Curves')
            plt.legend()
            curve_path = os.path.join(temp_dir, 'learning_curves.png')
            plt.savefig(curve_path)
            mlflow.log_artifact(curve_path, "visualizations")
            
            # 2. Matriz de Confusão
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            cm_path = os.path.join(temp_dir, 'confusion_matrix.png')
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path, "visualizations")
            
            # 3. Pacote do Modelo (MLmodel, etc)
            print("📦 Salvando pacote do modelo PyTorch...")
            model_package_path = os.path.join(temp_dir, "pytorch_model")
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="CV_Professional_Model"
            )
            
            # Limpeza
            shutil.rmtree(temp_dir)
            print(f"✅ Treino concluído! Melhor Acurácia: {best_acc:.4f}")
            print(f"🔗 Veja no DagsHub: https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow")

if __name__ == "__main__":
    # Exemplo de uso para Visão Computacional (Classificação)
    print("🚀 Iniciando MLOps para Visão Computacional...")
    mlops = DLCVProfessional()
    
    # Exemplo com dados sintéticos ou um dataset pequeno (ex: CIFAR10)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Criando um dataset de exemplo rápido (apenas para demonstrar o fluxo)
    # Em um projeto real, você usaria seus dados de imagem aqui
    try:
        print("📥 Carregando dataset CIFAR10 (exemplo)...")
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # Reduzir para teste rápido
        train_set = torch.utils.data.Subset(train_set, range(200)) 
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        
        val_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        val_set = torch.utils.data.Subset(val_set, range(50))
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        
        model = mlops.get_transfer_learning_model(model_name='resnet18', num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        mlops.train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=2)
        
    except Exception as e:
        print(f"❌ Erro durante o exemplo: {e}")
