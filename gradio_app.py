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
        if report_path:
            return f"📉 Relatório de Drift gerado: {report_path}"
        else:
            return "⚠️ Falha ao gerar relatório: Evidently AI não está instalado ou configurado corretamente."
    except Exception as e:
        return f"💥 Erro na análise de drift: {str(e)}"

def run_face_detection(image):
    if image is None:
        return None, "❌ Por favor, envie uma imagem."
    try:
        # Salvar imagem temporária para o YOLO
        temp_path = "temp_face_input.jpg"
        image.save(temp_path)
        res_path = mlops.detect_faces(temp_path)
        return res_path, "✅ Detecção Facial Concluída!"
    except Exception as e:
        return None, f"💥 Erro na detecção: {str(e)}"

def run_image_recommendation(query_img, gallery_folder):
    if query_img is None or not gallery_folder:
        return None, "❌ Forneça uma imagem e o caminho do diretório da galeria."
    try:
        temp_query = "temp_query.jpg"
        query_img.save(temp_query)
        results = mlops.recommend_similar(temp_query, gallery_folder)
        # Retornar apenas os caminhos das imagens
        return [r[0] for r in results], "✅ Recomendações geradas!"
    except Exception as e:
        return None, f"💥 Erro na recomendação: {str(e)}"

def run_nlp_analysis(text, task):
    if not text:
        return "❌ Por favor, insira um texto."
    try:
        result = mlops.nlp_analyze(text, task)
        return str(result)
    except Exception as e:
        return f"💥 Erro no NLP: {str(e)}"

def run_timeseries(file, date_col, target_col, periods):
    if file is None: return None, "❌ Envie um CSV."
    try:
        forecast, plot_path = mlops.train_timeseries(file.name, date_col, target_col, int(periods))
        return plot_path, "✅ Previsão concluída!"
    except Exception as e:
        return None, f"💥 Erro: {str(e)}"

def run_unsupervised(file, method, n_clusters):
    if file is None: return None, "❌ Envie um CSV."
    try:
        if method == "Clustering (K-Means)":
            res = mlops.train_clustering(file.name, n_clusters=int(n_clusters))
        else:
            res = mlops.detect_anomalies(file.name)
        return res, "✅ Processamento concluído!"
    except Exception as e:
        return None, f"💥 Erro: {str(e)}"

def run_finetuning(file, model_name, text_col, label_col):
    if file is None: return "❌ Envie um CSV."
    try:
        path = mlops.fine_tune_nlp(model_name, file.name, text_col, label_col)
        return f"✅ Fine-tuning concluído! Modelo salvo em: {path}"
    except Exception as e:
        return f"💥 Erro: {str(e)}"

# Interface Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="MLOps Enterprise Dashboard") as demo:
    gr.Markdown("""
    # 🎯 MLOps Enterprise Dashboard (V5.0)
    ### Framework Universal: AutoML, CV, NLP, Time Series e Fine-Tuning
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

    with gr.Tab("📈 Time Series"):
        with gr.Row():
            ts_input = gr.File(label="Dataset Temporal (CSV)")
            with gr.Column():
                date_col = gr.Textbox(label="Coluna de Data (ex: ds)", value="ds")
                target_col = gr.Textbox(label="Coluna Target (ex: y)", value="y")
                periods = gr.Number(label="Períodos de Previsão", value=30)
                ts_btn = gr.Button("🔮 Prever Futuro")
        ts_plot = gr.Image(label="Gráfico de Previsão")
        ts_status = gr.Textbox(label="Status")
        ts_btn.click(run_timeseries, inputs=[ts_input, date_col, target_col, periods], outputs=[ts_plot, ts_status])

    with gr.Tab("💎 Unsupervised"):
        with gr.Row():
            uns_input = gr.File(label="Dataset (CSV)")
            with gr.Column():
                uns_method = gr.Dropdown(["Clustering (K-Means)", "Anomaly Detection"], label="Método", value="Clustering (K-Means)")
                n_clusters = gr.Slider(2, 10, 3, label="Nº Clusters (se aplicável)")
                uns_btn = gr.Button("⚙️ Executar")
        uns_file = gr.File(label="Resultado (CSV)")
        uns_status = gr.Textbox(label="Status")
        uns_btn.click(run_unsupervised, inputs=[uns_input, uns_method, n_clusters], outputs=[uns_file, uns_status])

    with gr.Tab("🧠 Fine-Tuning"):
        with gr.Row():
            ft_input = gr.File(label="Dataset de Treino (CSV)")
            with gr.Column():
                ft_model = gr.Textbox(label="Modelo Base (HF)", value="distilbert-base-uncased")
                ft_text = gr.Textbox(label="Coluna de Texto", value="text")
                ft_label = gr.Textbox(label="Coluna de Label", value="label")
                ft_btn = gr.Button("🚀 Iniciar Fine-Tuning")
        ft_status = gr.Textbox(label="Status do Treino")
        ft_btn.click(run_finetuning, inputs=[ft_input, ft_model, ft_text, ft_label], outputs=[ft_status])

    with gr.Tab("📸 Computer Vision"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👤 Detecção Facial")
                face_input = gr.Image(type="pil", label="Upload Imagem")
                face_btn = gr.Button("🔍 Detectar Faces")
                face_status = gr.Textbox(label="Status")
                face_output = gr.Image(label="Resultado da Detecção")
            
            with gr.Column():
                gr.Markdown("### 🖼️ Recomendação de Imagens")
                query_input = gr.Image(type="pil", label="Imagem de Busca")
                gallery_path = gr.Textbox(label="Caminho da Galeria (Diretório)", placeholder="C:/fotos/galeria")
                rec_btn = gr.Button("🔎 Buscar Semelhantes")
                rec_status = gr.Textbox(label="Status")
                rec_gallery = gr.Gallery(label="Imagens Recomendadas")

        face_btn.click(run_face_detection, inputs=[face_input], outputs=[face_output, face_status])
        rec_btn.click(run_image_recommendation, inputs=[query_input, gallery_path], outputs=[rec_gallery, rec_status])

    with gr.Tab("📝 NLP & Sentiment"):
        with gr.Row():
            nlp_text = gr.Textbox(label="Texto para Análise", lines=5)
            nlp_task = gr.Dropdown(["sentiment-analysis", "summarization", "ner"], label="Tarefa", value="sentiment-analysis")
        nlp_btn = gr.Button("⚙️ Processar Texto")
        nlp_output = gr.Textbox(label="Resultado NLP")
        
        nlp_btn.click(run_nlp_analysis, inputs=[nlp_text, nlp_task], outputs=[nlp_output])

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
