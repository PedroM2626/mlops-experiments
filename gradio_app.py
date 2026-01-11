import gradio as gr
import pandas as pd
import os
from train_and_save_professional import MLOpsEnterprise
import logging

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializa o framework
mlops = MLOpsEnterprise()

def run_training(file_obj, task, engine, timeout):
    if file_obj is None:
        return "❌ Por favor, faça o upload de um arquivo CSV.", None, None
    
    try:
        # Salva o arquivo temporariamente para processamento
        file_path = file_obj.name
        df = pd.read_csv(file_path)
        
        # Executa o treino
        logger.info(f"Iniciando treino via Interface: {engine} ({task})")
        model, score = mlops.train_automl(file_path, task=task, engine=engine, timeout=int(timeout))
        
        if model is None:
            return f"❌ Erro: Engine {engine} não disponível.", None, None
            
        # Gera explicações SHAP (primeiras 100 linhas)
        X_train = df.drop(columns=[df.columns[-1]])
        mlops.explain_model(model, X_train, method='shap')
        
        status_msg = f"✅ Treino Concluído!\nEngine: {engine.upper()}\nScore: {score:.4f}"
        
        shap_plot = "shap_summary.png" if os.path.exists("shap_summary.png") else None
        
        # Gera API de Serving
        mlops.generate_serving_api("model.onnx")
        api_script = "serving_api.py" if os.path.exists("serving_api.py") else None
        
        return status_msg, shap_plot, api_script

    except Exception as e:
        logger.error(f"Erro na interface: {e}")
        return f"💥 Erro fatal: {str(e)}", None, None

def run_drift_analysis(ref_file, cur_file):
    if ref_file is None or cur_file is None:
        return "❌ Por favor, envie os dois arquivos (Referência e Atual)."
    
    try:
        ref_df = pd.read_csv(ref_file.name)
        cur_df = pd.read_csv(cur_file.name)
        
        report_path = mlops.detect_drift(ref_df, cur_df)
        return f"📉 Relatório de Drift gerado: {report_path}"
    except Exception as e:
        return f"💥 Erro na análise de drift: {str(e)}"

# Interface Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="MLOps Enterprise Dashboard") as demo:
    gr.Markdown("""
    # 🎯 MLOps Enterprise Dashboard (V4.0)
    ### Interface Visual para AutoML, Validação e Monitoramento
    """)
    
    with gr.Tab("🚀 AutoML Training"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload Dataset (CSV)", file_types=[".csv"])
                task_radio = gr.Radio(["classification", "regression"], label="Tipo de Tarefa", value="classification")
                engine_dropdown = gr.Dropdown(["flaml", "autogluon", "tpot", "autosklearn", "h2o"], label="Engine de AutoML", value="flaml")
                timeout_slider = gr.Slider(minimum=30, maximum=3600, value=60, step=30, label="Timeout (segundos)")
                train_btn = gr.Button("🔥 Iniciar Treinamento", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(label="Status do Processamento")
                shap_output = gr.Image(label="SHAP Feature Importance")
                api_output = gr.File(label="Serving API Script (Download)")
        
        train_btn.click(
            run_training, 
            inputs=[file_input, task_radio, engine_dropdown, timeout_slider], 
            outputs=[status_output, shap_output, api_output]
        )

    with gr.Tab("📉 Data Drift Analysis"):
        with gr.Row():
            ref_input = gr.File(label="Dataset de Referência (Treino)", file_types=[".csv"])
            cur_input = gr.File(label="Dataset Atual (Produção)", file_types=[".csv"])
        drift_btn = gr.Button("🔍 Analisar Drift", variant="secondary")
        drift_output = gr.Textbox(label="Resultado da Análise")
        
        drift_btn.click(
            run_drift_analysis,
            inputs=[ref_input, cur_input],
            outputs=[drift_output]
        )

    gr.Markdown("""
    ---
    **Recursos integrados:** MLflow Tracking, DagsHub, SHAP, ONNX Export, Evidently AI.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
