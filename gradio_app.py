import gradio as gr
import pandas as pd
import os

# Forçar Transformers a usar PyTorch e evitar conflitos com Keras 3 no Python 3.13
os.environ["USE_TF"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from train_and_save_professional import MLOpsEnterprise
import logging

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializa o framework
mlops = MLOpsEnterprise()

def get_sys_info():
    status = mlops.get_system_status()
    info = f"🖥️ **GPU**: {status['GPU_Device']} (Ativa: {status['GPU_Available']})\n"
    info += f"🔗 **MLflow**: {status['MLflow_URI']}\n"
    info += f"📦 **DagsHub**: {'✅ Conectado' if status['DagsHub_Connected'] else '❌ Desconectado'}\n"
    info += f"📊 **WandB**: {'✅ Conectado' if status['WandB_Connected'] else '❌ Desconectado'}\n\n"
    info += "**Engines Disponíveis:**\n"
    for eng, available in status['Engines'].items():
        info += f"- {eng}: {'✅' if available else '❌'}\n"
    return info

# --- HELPERS PARA INTERFACE DINÂMICA ---
def update_columns(file):
    if file is None:
        return gr.Dropdown(choices=[]), gr.Dropdown(choices=[])
    try:
        df = pd.read_csv(file.name)
        cols = df.columns.tolist()
        return gr.Dropdown(choices=cols, value=cols[-1]), gr.Dropdown(choices=cols, value=cols[0])
    except:
        return gr.Dropdown(choices=[]), gr.Dropdown(choices=[])

def run_standalone_validation(file, target):
    if file is None: return "❌ Envie um CSV."
    try:
        df = pd.read_csv(file.name)
        df_clean = mlops.validate_data(df, target)
        return f"✅ Dados Válidos!\nLinhas originais: {len(df)}\nLinhas após limpeza: {len(df_clean)}\nColunas: {len(df.columns)}\nTarget: {target}"
    except Exception as e:
        return f"❌ Erro na validação: {str(e)}"

def run_training(file_obj, target, task, engine, timeout, explain_method):
    if file_obj is None:
        return "❌ Por favor, faça o upload de um arquivo CSV.", None, None, None
    
    try:
        file_path = file_obj.name
        df = pd.read_csv(file_path)
        
        logger.info(f"Iniciando treino via Interface: {engine} ({task}) | Target: {target}")
        model, score = mlops.train_automl(file_path, target=target, task=task, engine=engine, timeout=int(timeout))
        
        if model is None:
            return f"❌ Erro: Engine {engine} não disponível.", None, None, None
            
        X_train = df.drop(columns=[target])
        exp_path = mlops.explain_model(model, X_train, method=explain_method.lower())
        
        status_msg = f"✅ Treino Concluído!\nEngine: {engine.upper()}\nScore: {score:.4f}"
        
        shap_plot = exp_path if explain_method == "SHAP" and exp_path and exp_path.endswith('.png') else None
        lime_file = exp_path if explain_method == "LIME" and exp_path and exp_path.endswith('.html') else None
        
        mlops.generate_serving_api("model.onnx")
        api_script = "serving_api.py" if os.path.exists("serving_api.py") else None
        
        return status_msg, shap_plot, lime_file, api_script

    except Exception as e:
        logger.error(f"Erro na interface: {e}")
        return f"💥 Erro fatal: {str(e)}", None, None, None

def run_advanced_training(file, target, mode, model_type, use_optuna, task, epochs):
    if file is None: return "❌ Envie um CSV.", None
    try:
        if mode == "Manual (Sklearn/XGB)":
            model, score = mlops.train_manual(file.name, target, model_type, use_optuna, task)
            msg = f"✅ Treino Manual Concluído!\nModelo: {model_type}\nScore: {score:.4f}"
        else:
            model, msg = mlops.train_pytorch_tabular(file.name, target, epochs=int(epochs), task=task)
        
        return msg, "model.onnx" if os.path.exists("model.onnx") else None
    except Exception as e:
        return f"💥 Erro: {str(e)}", None

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

def run_object_detection(image, task="generic"):
    if image is None:
        return None, "❌ Por favor, envie uma imagem."
    try:
        temp_path = f"temp_{task}_input.jpg"
        image.save(temp_path)
        res_path = mlops.detect_objects(temp_path, task=task)
        return res_path, f"✅ Detecção ({task}) Concluída!"
    except Exception as e:
        return None, f"💥 Erro na detecção: {str(e)}"

def run_zenml_pipeline_ui(file):
    if file is None: return "❌ Envie um CSV."
    try:
        res = mlops.run_zenml_pipeline(file.name)
        return f"✅ {res}"
    except Exception as e:
        return f"💥 Erro no ZenML: {str(e)}"

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
    # 🎯 MLOps Enterprise Dashboard (V7.0)
    ### O Framework Universal de IA e MLOps
    """)
    
    with gr.Tab("⚙️ System Status"):
        sys_info = gr.Markdown(get_sys_info())
        refresh_btn = gr.Button("🔄 Atualizar Status")
        refresh_btn.click(get_sys_info, outputs=[sys_info])
    
    with gr.Tab("🛤️ ZenML Pipelines"):
        gr.Markdown("### 🚀 Orquestração com ZenML")
        zen_input = gr.File(label="Dataset para Pipeline (CSV)")
        zen_btn = gr.Button("Executar Pipeline Completo")
        zen_output = gr.Textbox(label="Status do Pipeline")
        zen_btn.click(run_zenml_pipeline_ui, inputs=[zen_input], outputs=[zen_output])

    with gr.Tab("📊 Data Lab"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔍 Validação de Dados")
                val_input = gr.File(label="Dataset (CSV)")
                val_target = gr.Dropdown(label="Target Column", choices=[])
                val_btn = gr.Button("Validar Integridade")
                val_output = gr.Textbox(label="Resultado da Validação")
                val_btn.click(run_standalone_validation, inputs=[val_input, val_target], outputs=[val_output])
            
            with gr.Column():
                gr.Markdown("### 📉 Data Drift Analysis")
                ref_input = gr.File(label="Dataset de Referência (Treino)")
                cur_input = gr.File(label="Dataset Atual (Produção)")
                drift_btn = gr.Button("🔍 Analisar Drift")
                drift_output = gr.Textbox(label="Resultado da Análise")
                drift_btn.click(run_drift_analysis, inputs=[ref_input, cur_input], outputs=[drift_output])

    with gr.Tab("🚀 AutoML Training"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload Dataset (CSV)", file_types=[".csv"])
                target_dropdown = gr.Dropdown(label="Target Column", choices=[])
                task_radio = gr.Radio(["classification", "regression"], label="Tipo de Tarefa", value="classification")
                engine_dropdown = gr.Dropdown(["flaml", "autogluon", "tpot", "autosklearn", "h2o", "unified"], label="Engine de AutoML", value="flaml")
                timeout_slider = gr.Slider(minimum=30, maximum=3600, value=60, step=30, label="Timeout (segundos)")
                explain_radio = gr.Radio(["SHAP", "LIME"], label="Método de Explicabilidade", value="SHAP")
                train_btn = gr.Button("🔥 Iniciar Treinamento", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(label="Status do Processamento")
                shap_output = gr.Image(label="SHAP Summary (Plot)")
                lime_output = gr.File(label="LIME Explanation (HTML)")
                api_output = gr.File(label="Serving API Script (Download)")
        
        # Atualizar colunas ao carregar arquivo
        file_input.change(update_columns, inputs=[file_input], outputs=[target_dropdown, val_target])

        train_btn.click(
            run_training, 
            inputs=[file_input, target_dropdown, task_radio, engine_dropdown, timeout_slider, explain_radio], 
            outputs=[status_output, shap_output, lime_output, api_output]
        )

    with gr.Tab("�️ Advanced Training"):
        with gr.Row():
            with gr.Column():
                adv_file = gr.File(label="Dataset (CSV)")
                adv_target = gr.Dropdown(label="Target Column", choices=[])
                adv_mode = gr.Radio(["Manual (Sklearn/XGB)", "Deep Learning (PyTorch)"], label="Modo", value="Manual (Sklearn/XGB)")
                adv_model = gr.Dropdown(["rf", "xgb"], label="Modelo (se Manual)", value="rf")
                adv_optuna = gr.Checkbox(label="Otimizar com Optuna", value=False)
                adv_task = gr.Radio(["classification", "regression"], label="Tarefa", value="classification")
                adv_epochs = gr.Slider(5, 100, 10, step=5, label="Epochs (se DL)")
                adv_btn = gr.Button("🚀 Treinar", variant="primary")
            
            with gr.Column():
                adv_status = gr.Textbox(label="Status")
                adv_onnx = gr.File(label="Modelo Exportado (ONNX)")

        adv_file.change(update_columns, inputs=[adv_file], outputs=[adv_target, val_target])
        adv_btn.click(run_advanced_training, 
                     inputs=[adv_file, adv_target, adv_mode, adv_model, adv_optuna, adv_task, adv_epochs], 
                     outputs=[adv_status, adv_onnx])

    with gr.Tab("📈 Time Series"):
        with gr.Row():
            ts_input = gr.File(label="Dataset Temporal (CSV)")
            ts_date_col = gr.Dropdown(label="Coluna de Data (ds)", choices=[])
            ts_target_col = gr.Dropdown(label="Coluna Target (y)", choices=[])
            with gr.Column():
                periods = gr.Number(label="Períodos de Previsão", value=30)
                ts_btn = gr.Button("🔮 Prever Futuro")
        
        ts_input.change(update_columns, inputs=[ts_input], outputs=[ts_target_col, ts_date_col])
        
        ts_plot = gr.Image(label="Gráfico de Previsão")
        ts_status = gr.Textbox(label="Status")
        ts_btn.click(run_timeseries, inputs=[ts_input, ts_date_col, ts_target_col, periods], outputs=[ts_plot, ts_status])

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
                gr.Markdown("### 👤 Detecção (YOLOv8)")
                cv_input = gr.Image(type="pil", label="Upload Imagem ou Webcam")
                cv_task = gr.Radio(["faces", "generic"], label="Tipo de Detecção", value="generic")
                cv_btn = gr.Button("🔍 Detectar")
                cv_status = gr.Textbox(label="Status")
                cv_output = gr.Image(label="Resultado da Detecção")
                cv_btn.click(run_object_detection, inputs=[cv_input, cv_task], outputs=[cv_output, cv_status])
            
            with gr.Column():
                gr.Markdown("### 🖼️ Recomendação de Imagens")
                query_input = gr.Image(type="pil", label="Imagem de Busca")
                gallery_path = gr.Textbox(label="Caminho da Galeria (Diretório)", placeholder="C:/fotos/galeria")
                rec_btn = gr.Button("🔎 Buscar Semelhantes")
                rec_status = gr.Textbox(label="Status")
                rec_gallery = gr.Gallery(label="Imagens Recomendadas")
                rec_btn.click(run_image_recommendation, inputs=[query_input, gallery_path], outputs=[rec_gallery, rec_status])

        with gr.Row():
            gr.Markdown("### 🎥 Detecção em Tempo Real (Webcam)")
            rt_input = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Webcam Stream")
            rt_output = gr.Image(label="Annotated Stream")
            
            # Função interna para streaming real-time
            def stream_yolo(img):
                if img is None: return None
                temp_path = "temp_stream.jpg"
                img.save(temp_path)
                res_path = mlops.detect_objects(temp_path, task="generic")
                return res_path

            rt_input.stream(stream_yolo, inputs=[rt_input], outputs=[rt_output])

    with gr.Tab("📝 NLP & Sentiment"):
        with gr.Row():
            nlp_text = gr.Textbox(label="Texto para Análise", lines=5)
            nlp_task = gr.Dropdown(["sentiment-analysis", "summarization", "ner"], label="Tarefa", value="sentiment-analysis")
        nlp_btn = gr.Button("⚙️ Processar Texto")
        nlp_output = gr.Textbox(label="Resultado NLP")
        
        nlp_btn.click(run_nlp_analysis, inputs=[nlp_text, nlp_task], outputs=[nlp_output])

    # Remover Tab original de Drift pois agora está no Data Lab

    gr.Markdown("""
    ---
    **Recursos integrados:** MLflow Tracking, DagsHub, SHAP, ONNX Export, Evidently AI.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
