"""
Experimento 7: Multi-Task Learning (MMoE + TF-IDF)
==================================================================
Revertido para Embeddings Esparsos (TF-IDF) visando velocidade na CPU,
enquanto o treinamento Multi-Task (PyTorch) permanece acelerado por GPU.
Utiliza metricas F1-Weighted para balanceamento final de avaliacao.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.pytorch
import dagshub
from dotenv import load_dotenv
import warnings
from sklearn.metrics import f1_score
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

# Configuracao DagsHub / MLflow
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")
dagshub.init(repo_owner=repo_owner, repo_name=repo_name)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ============================================================================
# ARQUITETURAS NEURAIS
# ============================================================================

class SingleTaskModel(nn.Module):
    def __init__(self, input_dim):
        super(SingleTaskModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class MMoE_MultiTaskModel(nn.Module):
    def __init__(self, input_dim, num_experts=3):
        super(MMoE_MultiTaskModel, self).__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(num_experts)
        ])
        self.gate_joy = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=1))
        self.gate_sad = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=1))
        self.gate_ang = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=1))
        
        self.head_joy = nn.Linear(256, 1)
        self.head_sad = nn.Linear(256, 1)
        self.head_ang = nn.Linear(256, 1)

    def forward(self, x):
        exp_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        w_joy = self.gate_joy(x).unsqueeze(2)
        w_sad = self.gate_sad(x).unsqueeze(2)
        w_ang = self.gate_ang(x).unsqueeze(2)
        
        repr_joy = torch.sum(exp_outs * w_joy, dim=1)
        repr_sad = torch.sum(exp_outs * w_sad, dim=1)
        repr_ang = torch.sum(exp_outs * w_ang, dim=1)
        
        out_joy = self.head_joy(repr_joy).squeeze(1)
        out_sad = self.head_sad(repr_sad).squeeze(1)
        out_ang = self.head_ang(repr_ang).squeeze(1)
        
        return out_joy, out_sad, out_ang

# ============================================================================
# PIPELINE DE EXPERIMENTO
# ============================================================================

def run_experiment():
    print("\n" + "="*80)
    print("[INICIO] EXPERIMENTO 7: MMoE + TF-IDF + LOSS WEIGHTS + F1-WEIGHTED")
    print("="*80 + "\n")
    
    mlflow.set_experiment("MultiTask_Learning_V7_TFIDF_GPU")
    
    with mlflow.start_run(run_name="mmoe_tfidf_gpu_full_weighted"):
        print("[1] Baixando Dataset GoEmotions (Completo)...")
        dataset = load_dataset("go_emotions", "simplified")
        
        # Carregando todas as 43k amostras para maximo aproveitamento do Hardware
        df = dataset['train'].to_pandas()
        df_test = dataset['test'].to_pandas()
        
        label_names = dataset['train'].features['labels'].feature.names
        joy_idx = label_names.index('joy')
        sadness_idx = label_names.index('sadness')
        anger_idx = label_names.index('anger')
        
        df['y_joy'] = df['labels'].apply(lambda x: 1 if joy_idx in x else 0)
        df['y_sad'] = df['labels'].apply(lambda x: 1 if sadness_idx in x else 0)
        df['y_ang'] = df['labels'].apply(lambda x: 1 if anger_idx in x else 0)
        
        df_test['y_joy'] = df_test['labels'].apply(lambda x: 1 if joy_idx in x else 0)
        df_test['y_sad'] = df_test['labels'].apply(lambda x: 1 if sadness_idx in x else 0)
        df_test['y_ang'] = df_test['labels'].apply(lambda x: 1 if anger_idx in x else 0)
        
        print("[2] Extraindo Features com TF-IDF (CPU)...")
        # Usaremos TF-IDF para simplificar a etapa de extracao de features (Reversao Arquitetural)
        vectorizer = TfidfVectorizer(max_features=15000, stop_words='english')
        X_tr = vectorizer.fit_transform(df['text']).toarray()
        X_te = vectorizer.transform(df_test['text']).toarray()
        
        print(f"    -> Shape Treino: {X_tr.shape} | Shape Teste: {X_te.shape}")
        
        print("[3] Configurando PyTorch e GPU...")
        # Habilitar GPU (CUDA) se disponivel para o treinamento
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    -> Hardware Ativado: {device}")
        
        # Calculando Pesos para classes minoritarias (Loss Weighting)
        pos_w_joy = torch.tensor([(len(df) - df['y_joy'].sum()) / max(1, df['y_joy'].sum())]).to(device)
        pos_w_sad = torch.tensor([(len(df) - df['y_sad'].sum()) / max(1, df['y_sad'].sum())]).to(device)
        pos_w_ang = torch.tensor([(len(df) - df['y_ang'].sum()) / max(1, df['y_ang'].sum())]).to(device)
        
        # Criterios dedicados com pesos
        crit_joy = nn.BCEWithLogitsLoss(pos_weight=pos_w_joy)
        crit_sad = nn.BCEWithLogitsLoss(pos_weight=pos_w_sad)
        crit_ang = nn.BCEWithLogitsLoss(pos_weight=pos_w_ang)
        
        # Tensores
        X_train_t = torch.FloatTensor(X_tr).to(device)
        y_joy_train_t = torch.FloatTensor(df['y_joy'].values).to(device)
        y_sad_train_t = torch.FloatTensor(df['y_sad'].values).to(device)
        y_ang_train_t = torch.FloatTensor(df['y_ang'].values).to(device)
        
        X_test_t = torch.FloatTensor(X_te).to(device)
        
        dataset_train = TensorDataset(X_train_t, y_joy_train_t, y_sad_train_t, y_ang_train_t)
        loader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
        
        input_dim = X_tr.shape[1] # 5000 do TFIDF
        epochs = 12
        
        # ================================================================
        # TREINAMENTO SINGLE-TASK
        # ================================================================
        print("\n[4] Treinando Modelos Single-Task (Isolados)...")
        model_st_joy = SingleTaskModel(input_dim).to(device)
        model_st_sad = SingleTaskModel(input_dim).to(device)
        model_st_ang = SingleTaskModel(input_dim).to(device)
        
        opt_st_joy = optim.Adam(model_st_joy.parameters(), lr=0.001)
        opt_st_sad = optim.Adam(model_st_sad.parameters(), lr=0.001)
        opt_st_ang = optim.Adam(model_st_ang.parameters(), lr=0.001)
        
        # LR Schedulers (Cai pela metade a cada 4 epocas)
        sch_j = optim.lr_scheduler.StepLR(opt_st_joy, step_size=4, gamma=0.5)
        sch_s = optim.lr_scheduler.StepLR(opt_st_sad, step_size=4, gamma=0.5)
        sch_a = optim.lr_scheduler.StepLR(opt_st_ang, step_size=4, gamma=0.5)
        
        for epoch in range(epochs):
            model_st_joy.train(); model_st_sad.train(); model_st_ang.train()
            for bx, by_j, by_s, by_a in loader_train:
                opt_st_joy.zero_grad()
                loss_j = crit_joy(model_st_joy(bx), by_j)
                loss_j.backward()
                opt_st_joy.step()
                
                opt_st_sad.zero_grad()
                loss_s = crit_sad(model_st_sad(bx), by_s)
                loss_s.backward()
                opt_st_sad.step()
                
                opt_st_ang.zero_grad()
                loss_a = crit_ang(model_st_ang(bx), by_a)
                loss_a.backward()
                opt_st_ang.step()
            
            sch_j.step(); sch_s.step(); sch_a.step()
        
        # ================================================================
        # TREINAMENTO MMoE MULTI-TASK
        # ================================================================
        print("[5] Treinando Modelo MMoE Multi-Task (Joint Loss + Gates)...")
        model_mt = MMoE_MultiTaskModel(input_dim, num_experts=3).to(device)
        opt_mt = optim.Adam(model_mt.parameters(), lr=0.001)
        sch_mt = optim.lr_scheduler.StepLR(opt_mt, step_size=4, gamma=0.5)
        
        for epoch in range(epochs):
            model_mt.train()
            for bx, by_j, by_s, by_a in loader_train:
                opt_mt.zero_grad()
                out_j, out_s, out_a = model_mt(bx)
                
                loss_j = crit_joy(out_j, by_j)
                loss_s = crit_sad(out_s, by_s)
                loss_a = crit_ang(out_a, by_a)
                
                joint_loss = loss_j + loss_s + loss_a
                joint_loss.backward()
                opt_mt.step()
                
            sch_mt.step()
        
        # ================================================================
        # AVALIACAO E COMPARACAO
        # ================================================================
        print("\n[6] Avaliando Resultados na Base de Teste (F1-Weighted)...")
        model_st_joy.eval(); model_st_sad.eval(); model_st_ang.eval(); model_mt.eval()
        
        with torch.no_grad():
            pred_st_joy = (torch.sigmoid(model_st_joy(X_test_t)) > 0.5).int().cpu().numpy()
            pred_st_sad = (torch.sigmoid(model_st_sad(X_test_t)) > 0.5).int().cpu().numpy()
            pred_st_ang = (torch.sigmoid(model_st_ang(X_test_t)) > 0.5).int().cpu().numpy()
            
            out_mt_j, out_mt_s, out_mt_a = model_mt(X_test_t)
            pred_mt_joy = (torch.sigmoid(out_mt_j) > 0.5).int().cpu().numpy()
            pred_mt_sad = (torch.sigmoid(out_mt_s) > 0.5).int().cpu().numpy()
            pred_mt_ang = (torch.sigmoid(out_mt_a) > 0.5).int().cpu().numpy()
            
        y_test_joy = df_test['y_joy'].values
        y_test_sad = df_test['y_sad'].values
        y_test_ang = df_test['y_ang'].values
        
        f1_st_joy = f1_score(y_test_joy, pred_st_joy, average='weighted')
        f1_st_sad = f1_score(y_test_sad, pred_st_sad, average='weighted')
        f1_st_ang = f1_score(y_test_ang, pred_st_ang, average='weighted')
        avg_st = (f1_st_joy + f1_st_sad + f1_st_ang) / 3
        
        f1_mt_joy = f1_score(y_test_joy, pred_mt_joy, average='weighted')
        f1_mt_sad = f1_score(y_test_sad, pred_mt_sad, average='weighted')
        f1_mt_ang = f1_score(y_test_ang, pred_mt_ang, average='weighted')
        avg_mt = (f1_mt_joy + f1_mt_sad + f1_mt_ang) / 3
        
        improvement = ((avg_mt - avg_st) / avg_st) * 100
        
        print("\n--- SINGLE-TASK (F1-Score Weighted) ---")
        print(f"Alegria: {f1_st_joy:.4f} | Tristeza: {f1_st_sad:.4f} | Raiva: {f1_st_ang:.4f}")
        print(f"Media: {avg_st:.4f}")
        
        print("\n--- MMoE MULTI-TASK (F1-Score Weighted) ---")
        print(f"Alegria: {f1_mt_joy:.4f} | Tristeza: {f1_mt_sad:.4f} | Raiva: {f1_mt_ang:.4f}")
        print(f"Media: {avg_mt:.4f}")
        
        print(f"\n[!] GANHO DO MMoE: {improvement:+.2f}%")
        
        mlflow.log_params({
            "dataset": "go_emotions",
            "samples": len(df),
            "epochs": epochs,
            "architecture": "MMoE",
            "embeddings": "TF-IDF (5000 features)",
            "loss_weighting": "Enabled",
            "scheduler": "StepLR",
            "metric": "f1_weighted"
        })
        
        mlflow.log_metrics({
            "single_task_avg_f1_weighted": avg_st,
            "mmoe_avg_f1_weighted": avg_mt,
            "improvement_pct": improvement
        })
        
        mlflow.pytorch.log_model(model_mt, "mmoe_tfidf_model")
        
        print("\n[OK] MLOps concluido. Modelos baseados em TF-IDF salvos no DagsHub!")

if __name__ == "__main__":
    run_experiment()
