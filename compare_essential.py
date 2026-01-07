#!/usr/bin/env python3
"""
🎯 MLOPS ESSENCIAL - Script único para comparação de modelos
Compara dois modelos salvos usando MLflow + DagsHub
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def setup_mlflow():
    """Configura MLflow local (sem DagsHub)"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return mlflow

def load_model_safe(model_path):
    """Carrega modelo com tratamento de erro"""
    try:
        model = joblib.load(model_path)
        print(f"✅ Modelo carregado: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Erro ao carregar {model_path}: {e}")
        return None

def prepare_data(df_val, model_type):
    """Prepara dados baseado no tipo de modelo"""
    if model_type == 'knn':
        # KNN usa todas as features
        return df_val.drop(['sentiment'], axis=1)
    else:  # linearsvc
        # LinearSVC usa só texto
        return df_val['text_lemmatized'].fillna('')

def evaluate_model(model, X_val, y_val, model_name):
    """Avalia modelo e retorna métricas"""
    try:
        y_pred = model.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0)
        }
        
        print(f"\n📊 {model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    except Exception as e:
        print(f"❌ Erro ao avaliar {model_name}: {e}")
        return None

def main():
    """Função principal - comparação simples"""
    print("🎯 COMPARAÇÃO ESSENCIAL DE MODELOS")
    print("="*50)
    
    # Setup MLflow
    mlflow = setup_mlflow()
    
    # Configurar experimento
    mlflow.set_experiment("model_comparison_essential")
    
    # Carregar dados
    print("📁 Carregando dados...")
    try:
        df_val = pd.read_csv('processed_validation.csv')
        print(f"✅ Dataset de validação: {len(df_val)} amostras")
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return
    
    # Modelos para comparar
    models = {
        'knn': 'models/senti-pred-free-mlops-knn.pkl',
        'linearsvc': 'models/senti-pred-linearsvc.pkl'
    }
    
    results = {}
    
    # Comparar cada modelo
    for model_type, model_path in models.items():
        print(f"\n{'-'*40}")
        print(f"🔍 Avaliando {model_type.upper()}")
        print(f"{'-'*40}")
        
        # Carregar modelo
        model = load_model_safe(model_path)
        if not model:
            continue
        
        # Preparar dados
        X_val = prepare_data(df_val, model_type)
        y_val = df_val['sentiment']
        
        # Avaliar
        with mlflow.start_run(run_name=model_type):
            metrics = evaluate_model(model, X_val, y_val, model_type)
            
            if metrics:
                # Logar no MLflow
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("validation_samples", len(y_val))
                for metric, value in metrics.items():
                    mlflow.log_metric(metric, value)
                
                results[model_type] = metrics
    
    # Resultado final
    print(f"\n{'='*50}")
    print("🏆 RESULTADO FINAL")
    print(f"{'='*50}")
    
    if results:
        print("\n📊 Comparação direta:")
        print(f"{'Modelo':<12} {'Accuracy':<10} {'F1-Score':<10}")
        print("-" * 35)
        
        for model_type, metrics in results.items():
            print(f"{model_type:<12} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        # Vencedor
        winner = max(results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\n🥇 VENCEDOR: {winner[0].upper()} com F1-Score: {winner[1]['f1_score']:.4f}")
    else:
        print("❌ Nenhum modelo pôde ser avaliado")

if __name__ == "__main__":
    main()