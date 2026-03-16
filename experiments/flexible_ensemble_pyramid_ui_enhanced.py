import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Add the parent directory to the path to import the pyramid module
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from experiments.flexible_ensemble_pyramid import (
        PyramidEnsemble, 
        RLMetaLearner, 
        NASController,
        setup_tracking,
        load_data,
        clean_tweet,
        get_model,
        SEED,
        CV_FOLDS,
        EXP_NAME,
        PATIENCE,
        MIN_MODELS_PER_LAYER,
        MAX_MODELS_PER_LAYER,
        EPSILON_RL,
        OPTIM_METRIC,
        TFIDF_MAX,
        TFIDF_NGRAMS,
        JITTER,
        STRATEGY,
        mlflow
    )
except ImportError:
    st.error("Could not import flexible_ensemble_pyramid module. Please ensure it's in the correct location.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Flexible Ensemble Pyramid - Enhanced",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .layer-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .model-badge {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        margin: 0.3rem;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .connection-line {
        stroke: #6c757d;
        stroke-width: 2;
        stroke-dasharray: 5, 5;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>🧠 Enhanced Flexible Ensemble Pyramid</h1>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("<h2 style='color: #495057;'>⚙️ Configuração</h2>", unsafe_allow_html=True)
    
    # Speed Presets
    st.markdown("<h3 style='color: #6c757d;'>⚡ Modo de Velocidade</h3>", unsafe_allow_html=True)
    speed_mode = st.radio(
        "Selecione o modo:",
        ["Personalizado", "Relâmpago (Fastest)", "Equilibrado (Default)", "Completo (Slowest)"],
        index=2,
        help="Ajusta automaticamente os parâmetros para priorizar velocidade ou precisão."
    )
    
    # Default values based on speed mode
    if speed_mode == "Relâmpago (Fastest)":
        num_layers_val, min_models_val, max_models_val = 3, 2, 2
        tfidf_max_val, subsample_val = 5000, 5000
        use_nas_val, use_jitter_val = False, False
        strategy_val = "simple"
    elif speed_mode == "Equilibrado (Default)":
        num_layers_val, min_models_val, max_models_val = 6, 2, 4
        tfidf_max_val, subsample_val = 15000, 15000
        use_nas_val, use_jitter_val = True, True
        strategy_val = "residual"
    elif speed_mode == "Completo (Slowest)":
        num_layers_val, min_models_val, max_models_val = 12, 2, 6
        tfidf_max_val, subsample_val = 50000, 0 # 0 means full dataset
        use_nas_val, use_jitter_val = True, True
        strategy_val = "dense"
    else: # Personalizado
        num_layers_val, min_models_val, max_models_val = 12, 2, 6
        tfidf_max_val, subsample_val = 50000, 0
        use_nas_val, use_jitter_val = True, True
        strategy_val = "dense"

    # Experiment settings
    st.markdown("<h3 style='color: #6c757d;'>🔬 Configurações do Experimento</h3>", unsafe_allow_html=True)
    num_layers = st.slider("Número de Camadas", 2, 20, num_layers_val, 1,
                          help="Número de camadas hierárquicas na pirâmide")
    cv_folds = st.slider("CV Folds", 2, 10, 3, 1,
                        help="Número de dobras na validação cruzada")
    patience = st.slider("Paciência", 1, 10, 3, 1,
                        help="Paciência para early stopping")
    
    # Model selection parameters
    st.markdown("<h3 style='color: #6c757d;'>🤖 Seleção de Modelos</h3>", unsafe_allow_html=True)
    min_models = st.slider("Min Modelos por Camada", 1, 10, min_models_val, 1,
                          help="Número mínimo de modelos por camada")
    max_models = st.slider("Max Modelos por Camada", 2, 15, max_models_val, 1,
                          help="Número máximo de modelos por camada")
    epsilon_rl = st.slider("Taxa de Exploração RL", 0.0, 1.0, 0.2, 0.05,
                          help="Reinforcement learning exploration rate")
    
    # Feature engineering
    st.markdown("<h3 style='color: #6c757d;'>🔧 Engenharia de Features</h3>", unsafe_allow_html=True)
    tfidf_max = st.slider("TF-IDF Max Features", 1000, 100000, tfidf_max_val, 1000,
                         help="Número máximo de features TF-IDF")
    subsample_size = st.number_input("Tamanho da Amostra (0 = Tudo)", 0, 100000, subsample_val, 1000,
                                    help="Reduz o dataset para acelerar o treino.")
    tfidf_ngrams = st.selectbox("TF-IDF N-grams", [(1, 1), (1, 2), (1, 3), (2, 2)], index=1,
                               help="N-gram range for TF-IDF")
    
    # Advanced options
    st.markdown("<h3 style='color: #6c757d;'>⚡ Opções Avançadas</h3>", unsafe_allow_html=True)
    use_jitter = st.checkbox("Usar Hyperparameter Jitter", value=use_jitter_val,
                           help="Adiciona variação aleatória aos hiperparâmetros")
    use_nas = st.checkbox("Usar NAS Controller", value=use_nas_val,
                        help="Usa Neural Architecture Search para otimização")
    strategy = st.selectbox("Estratégia de Conexão", ["dense", "residual", "simple"], 
                           index=["dense", "residual", "simple"].index(strategy_val),
                           help="Como os modelos se conectam entre camadas")
    metric = st.selectbox("Métrica de Otimização", ["f1", "accuracy"], index=0,
                        help="Métrica principal para otimização")
    n_jobs = st.slider("Paralelismo (n_jobs)", -1, 16, -1, 1,
                      help="-1 usa todos os cores disponíveis. Aumentar acelera o treino.")
    
    # Visualization options
    st.markdown("<h3 style='color: #6c757d;'>🎨 Visualização</h3>", unsafe_allow_html=True)
    show_connections = st.checkbox("Show Model Connections", value=True,
                                 help="Display connections between models across layers")
    show_performance = st.checkbox("Show Performance Heatmap", value=True,
                                  help="Display performance heatmap overlay")
    
    # Run button
    if st.button("🚀 Run Ensemble Training", use_container_width=True, 
                help="Start the ensemble training process"):
        st.session_state.run_training = True

# Initialize session state
if 'run_training' not in st.session_state:
    st.session_state.run_training = False
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'pyramid' not in st.session_state:
    st.session_state.pyramid = None
if 'layer_strategy_map' not in st.session_state:
    st.session_state.layer_strategy_map = {}
if 'artifacts_dir' not in st.session_state:
    st.session_state.artifacts_dir = None

# Enhanced visualization functions
def create_enhanced_ensemble_visualization(results, pyramid, show_connections=True, show_performance=True, layer_strategy_map=None):
    """Create enhanced interactive visualization with tree branching"""
    
    if not results:
        return None
    
    # Prepare data for visualization
    layers_data = {}
    for result in results:
        layer_idx = result['layer']
        if layer_idx not in layers_data:
            layers_data[layer_idx] = []
        layers_data[layer_idx].append(result)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Colors for different model types with better contrast
    model_colors = {
        'lr': '#1f77b4', 'svc': '#ff7f0e', 'nb': '#2ca02c',
        'rf': '#d62728', 'et': '#9467bd', 'ridge': '#8c564b',
        'bag_lr': '#e377c2', 'bag_svc': '#7f7f7f', 'bag_nb': '#bcbd22'
    }
    
    # Position layers vertically with better spacing
    max_models_in_layer = max(len(models) for models in layers_data.values())
    layer_height = 2.0
    horizontal_spacing = 0.5
    
    # Store node positions for connection drawing
    node_positions = {}
    
    # Add nodes (models) with enhanced styling
    for layer_idx, models in layers_data.items():
        y_pos = -layer_idx * layer_height
        num_models = len(models)
        
        for i, model_result in enumerate(models):
            model_name = model_result['model']
            base_model = model_name.split('_')[0] if 'bag_' in model_name else model_name
            color = model_colors.get(base_model, '#17becf')
            
            # Calculate x position (centered with spacing)
            x_pos = (i - (num_models - 1) / 2) * horizontal_spacing
            
            # Store position for connections
            node_positions[(layer_idx, model_name)] = (x_pos, y_pos)
            
            # Node size based on performance
            node_size = 15 + (model_result['f1'] * 20)  # Scale size with F1 score
            
            # Add node with enhanced styling
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=color,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=model_name,
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                name=f"Layer {layer_idx}: {model_name}",
                customdata=[[
                    model_result['f1'], 
                    model_result['accuracy'], 
                    model_result['duration'],
                    base_model
                ]],
                hovertemplate="""
                <b>%{text}</b><br>
                Layer: %{customdata[3]}<br>
                F1: %{customdata[0]:.3f}<br>
                Accuracy: %{customdata[1]:.3f}<br>
                Duration: %{customdata[2]:.1f}s
                <extra></extra>
                """
            ))
    
    # Add enhanced connections between layers with tree branching
    if show_connections and len(layers_data) > 1:
        sorted_layers = sorted(layers_data.keys())
        for target_layer in sorted_layers:
            if target_layer <= 1:
                continue
            next_models = layers_data.get(target_layer, [])
            if not next_models:
                continue
            strategy_for_layer = "simple"
            if isinstance(layer_strategy_map, dict):
                strategy_for_layer = layer_strategy_map.get(target_layer, "simple")
            if strategy_for_layer == "dense":
                source_layers = [l for l in sorted_layers if l < target_layer]
            else:
                source_layers = [target_layer - 1] if (target_layer - 1) in layers_data else []
            for source_layer in source_layers:
                current_models = layers_data.get(source_layer, [])
                for current_model in current_models:
                    current_name = current_model['model']
                    current_pos = node_positions.get((source_layer, current_name))
                    if current_pos is None:
                        continue
                    for next_model in next_models:
                        next_name = next_model['model']
                        next_pos = node_positions.get((target_layer, next_name))
                        if next_pos is None:
                            continue
                        perf_similarity = 1 - abs(current_model['f1'] - next_model['f1'])
                        line_width = 1 + (perf_similarity * 3)
                        if next_model['f1'] > current_model['f1']:
                            line_color = 'green'
                        elif next_model['f1'] < current_model['f1']:
                            line_color = 'red'
                        else:
                            line_color = 'gray'
                        line_dash = "solid" if strategy_for_layer == "dense" else "dot"
                        fig.add_trace(go.Scatter(
                            x=[current_pos[0], next_pos[0]],
                            y=[current_pos[1], next_pos[1]],
                            mode='lines',
                            line=dict(width=line_width, color=line_color, dash=line_dash),
                            showlegend=False,
                            hoverinfo='none',
                            opacity=0.6
                        ))
    
    # Add performance heatmap overlay
    if show_performance:
        # Create a heatmap layer for performance visualization
        heatmap_data = []
        for layer_idx, models in layers_data.items():
            for i, model_result in enumerate(models):
                x_pos = (i - (len(models) - 1) / 2) * horizontal_spacing
                y_pos = -layer_idx * layer_height
                heatmap_data.append([x_pos, y_pos, model_result['f1']])
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, columns=['x', 'y', 'f1'])
            
            fig.add_trace(go.Contour(
                z=heatmap_df['f1'],
                x=heatmap_df['x'],
                y=heatmap_df['y'],
                colorscale='Viridis',
                showscale=True,
                opacity=0.3,
                name="Performance Heatmap",
                hoverinfo='none'
            ))
    
    # Enhanced layout configuration
    fig.update_layout(
        title=dict(
            text="Enhanced Ensemble Pyramid Architecture",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2c3e50')
        ),
        showlegend=True,
        hovermode='closest',
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-max_models_in_layer * horizontal_spacing / 2, 
                   max_models_in_layer * horizontal_spacing / 2]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-(len(layers_data) * layer_height) + layer_height, layer_height]
        ),
        height=800,
        margin=dict(l=20, r=20, t=80, b=20),
        plot_bgcolor='rgba(240,240,240,0.1)',
        paper_bgcolor='rgba(240,240,240,0.1)',
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig

def create_interactive_performance_dashboard(results):
    """Create interactive performance dashboard with multiple views"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    df['layer_model'] = df['layer'].astype(str) + '_' + df['model']
    df['model_type'] = df['model'].apply(lambda x: x.split('_')[0] if 'bag_' in x else x)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Trends", "📊 Model Distribution", "🔥 Heatmap", "📋 Details"])
    
    with tab1:
        # Performance trends across layers
        fig1 = px.line(df, x='layer', y='f1', color='model', 
                      title="F1 Score Evolution Across Layers",
                      labels={'layer': 'Layer', 'f1': 'F1 Score'})
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Accuracy trends
        fig2 = px.line(df, x='layer', y='accuracy', color='model',
                      title="Accuracy Evolution Across Layers")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Model type distribution
        fig3 = px.pie(df, names='model_type', 
                     title="Model Type Distribution",
                     hole=0.3)
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Performance by model type
        fig4 = px.box(df, x='model_type', y='f1', 
                     title="F1 Score Distribution by Model Type")
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        # Performance heatmap
        pivot_df = df.pivot(index='model', columns='layer', values='f1')
        fig5 = px.imshow(pivot_df, 
                        title="F1 Score Heatmap (Model × Layer)",
                        color_continuous_scale='Viridis')
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab4:
        # Detailed performance table
        st.dataframe(
            df[['layer', 'model', 'model_type', 'f1', 'accuracy', 'duration']]
            .sort_values(['layer', 'f1'], ascending=[True, False])
            .style.format({'f1': '{:.3f}', 'accuracy': '{:.3f}', 'duration': '{:.1f}'})
            .background_gradient(subset=['f1', 'accuracy'], cmap='YlGnBu')
            .set_properties(**{'text-align': 'center'})
        )

def create_layer_analysis_panel(results):
    """Create detailed layer analysis panel"""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Layer selector
    layers = sorted(df['layer'].unique())
    selected_layer = st.selectbox("Select Layer to Analyze", layers)
    
    layer_data = df[df['layer'] == selected_layer]
    best_model = layer_data.loc[layer_data['f1'].idxmax()]
    
    # Layer metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models in Layer", len(layer_data))
    with col2:
        st.metric("Best F1 Score", f"{best_model['f1']:.3f}")
    with col3:
        st.metric("Average F1", f"{layer_data['f1'].mean():.3f}")
    with col4:
        st.metric("Performance Range", f"{layer_data['f1'].max() - layer_data['f1'].min():.3f}")
    
    # Model cards
    st.markdown("### 🎯 Models in This Layer")
    cols = st.columns(3)
    
    for i, (_, model) in enumerate(layer_data.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                        border-radius: 1rem; padding: 1rem; margin: 0.5rem; 
                        border-left: 4px solid {'#28a745' if model['f1'] == best_model['f1'] else '#6c757d'};
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='margin: 0; color: #495057;'>{model['model']}</h4>
                <p style='margin: 0.5rem 0; color: #6c757d;'>
                    F1: <b>{model['f1']:.3f}</b><br>
                    Accuracy: <b>{model['accuracy']:.3f}</b><br>
                    Duration: <b>{model['duration']:.1f}s</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

# Training execution
if st.session_state.run_training:
    with st.spinner("🚀 Training ensemble pyramid..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        live_metrics_placeholder = st.empty()
        live_chart_placeholder = st.empty()
        live_results_placeholder = st.empty()

        try:
            status_text.text("📊 Setting up tracking...")
            setup_tracking()
            status_text.text("📊 Loading data...")
            train_df, val_df = load_data(subsample_train=subsample_size, subsample_val=subsample_size // 4 if subsample_size > 0 else 0)
            status_text.text("🔧 Vectorizing text...")
            vectorizer = TfidfVectorizer(
                max_features=tfidf_max,
                ngram_range=tfidf_ngrams,
                stop_words='english',
                min_df=2 # Speed up by ignoring very rare words
            )
            X_train = vectorizer.fit_transform(train_df['clean_text'].fillna(""))
            X_val = vectorizer.transform(val_df['clean_text'].fillna(""))
            le = LabelEncoder()
            y_train = le.fit_transform(train_df['sentiment'])
            y_val = le.transform(val_df['sentiment'])

            rl_learner = RLMetaLearner(epsilon=epsilon_rl, metric=metric)
            nas_controller = NASController(metric=metric) if use_nas else None
            pyramid = PyramidEnsemble(
                num_layers=num_layers,
                meta_learner=rl_learner,
                patience=patience,
                metric=metric,
                jitter=use_jitter,
                strategy=strategy,
                min_models=min_models,
                max_models=max_models,
                nas_controller=nas_controller,
                n_jobs=n_jobs
            )

            layer_strategy_map = {}
            training_counters = {"planned_models": 0, "trained_models": 0}

            def render_live_dashboard(current_results, current_status):
                if current_results:
                    done_layers = sorted({r["layer"] for r in current_results})
                    best_f1_live = max(r["f1"] for r in current_results)
                    best_acc_live = max(r["accuracy"] for r in current_results)
                    cols = live_metrics_placeholder.columns(4)
                    cols[0].metric("Modelos avaliados", len(current_results))
                    cols[1].metric("Camadas processadas", len(done_layers))
                    cols[2].metric("Melhor F1", f"{best_f1_live:.4f}")
                    cols[3].metric("Melhor Acurácia", f"{best_acc_live:.4f}")
                    live_fig = create_enhanced_ensemble_visualization(
                        current_results,
                        pyramid,
                        show_connections,
                        show_performance,
                        layer_strategy_map
                    )
                    if live_fig is not None:
                        live_chart_placeholder.plotly_chart(live_fig, use_container_width=True)
                    live_df = pd.DataFrame(current_results).sort_values(["layer", "f1"], ascending=[True, False])
                    live_results_placeholder.dataframe(
                        live_df[["layer", "model", "f1", "accuracy", "duration"]].head(20),
                        use_container_width=True
                    )
                status_text.text(current_status)

            def on_training_event(event):
                event_name = event.get("event")
                event_layer = event.get("layer")
                current_results = event.get("results", [])
                if event_name == "layer_config":
                    layer_strategy_map[event_layer] = event.get("strategy", strategy)
                    models_for_layer = event.get("models_to_run", [])
                    training_counters["planned_models"] += len(models_for_layer) + (1 if len(models_for_layer) > 1 else 0)
                    render_live_dashboard(
                        current_results,
                        f"🧭 Camada {event_layer}/{num_layers}: estratégia {layer_strategy_map[event_layer]} | modelos {models_for_layer}"
                    )
                elif event_name in {"model_trained", "voting_trained"}:
                    training_counters["trained_models"] += 1
                    progress_value = min(
                        1.0,
                        training_counters["trained_models"] / max(training_counters["planned_models"], 1)
                    )
                    progress_bar.progress(progress_value)
                    render_live_dashboard(
                        current_results,
                        f"⚙️ Camada {event_layer}/{num_layers}: treinou {event.get('model_name')}"
                    )
                elif event_name == "layer_start":
                    progress_layer = (event_layer - 1) / max(num_layers, 1)
                    progress_bar.progress(progress_layer)
                    render_live_dashboard(current_results, f"🏗️ Iniciando camada {event_layer}/{num_layers}")
                elif event_name == "layer_done":
                    progress_layer = event_layer / max(num_layers, 1)
                    progress_bar.progress(progress_layer)
                    render_live_dashboard(
                        current_results,
                        f"✅ Camada {event_layer}/{num_layers} concluída | features: {event.get('n_features')}"
                    )
                elif event_name == "training_done":
                    progress_bar.progress(1.0)
                    render_live_dashboard(current_results, "✅ Treinamento concluído")

            status_text.text("🧠 Treinando camadas com atualização em tempo real...")
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifacts_dir = Path("experiments") / "artifacts" / f"flexible_ensemble_ui_{run_stamp}"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            with mlflow.start_run(run_name=f"streamlit_ui_{run_stamp}"):
                mlflow.log_params({
                    "source": "streamlit_ui",
                    "num_layers": num_layers,
                    "patience": patience,
                    "min_models": min_models,
                    "max_models": max_models,
                    "epsilon_rl": epsilon_rl,
                    "metric": metric,
                    "strategy": strategy,
                    "jitter": use_jitter,
                    "use_nas": use_nas,
                    "tfidf_max": tfidf_max,
                    "tfidf_ngrams": str(tfidf_ngrams),
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                })

                pyramid.train(X_train, y_train, X_val, y_val, progress_callback=on_training_event)

                results_df = pd.DataFrame(pyramid.results)
                results_path = artifacts_dir / "pyramid_results.csv"
                results_df.to_csv(results_path, index=False)
                mlflow.log_artifact(str(results_path))

                config_payload = {
                    "num_layers": num_layers,
                    "patience": patience,
                    "min_models": min_models,
                    "max_models": max_models,
                    "epsilon_rl": epsilon_rl,
                    "metric": metric,
                    "strategy": strategy,
                    "jitter": use_jitter,
                    "use_nas": use_nas,
                    "tfidf_max": tfidf_max,
                    "tfidf_ngrams": list(tfidf_ngrams),
                    "layer_strategy_map": layer_strategy_map,
                    "best_score": float(pyramid.best_score),
                }
                config_path = artifacts_dir / "run_config.json"
                with config_path.open("w", encoding="utf-8") as f:
                    json.dump(config_payload, f, indent=2)
                mlflow.log_artifact(str(config_path))

                vectorizer_path = artifacts_dir / "tfidf_vectorizer.pkl"
                joblib.dump(vectorizer, vectorizer_path)
                mlflow.log_artifact(str(vectorizer_path))

                encoder_path = artifacts_dir / "label_encoder.pkl"
                joblib.dump(le, encoder_path)
                mlflow.log_artifact(str(encoder_path))

                if rl_learner.path.exists():
                    mlflow.log_artifact(str(rl_learner.path))
                if nas_controller and nas_controller.path.exists():
                    mlflow.log_artifact(str(nas_controller.path))
                if pyramid.best_model is not None:
                    best_model_path = artifacts_dir / "best_pyramid_model.pkl"
                    joblib.dump(pyramid.best_model, best_model_path)
                    mlflow.log_artifact(str(best_model_path))
                    best_res = max(
                        pyramid.results,
                        key=lambda r: r["f1"] if metric == "f1" else r["accuracy"]
                    )
                    mlflow.log_params({"best_model": best_res["model"], "best_layer": int(best_res["layer"])})
                    mlflow.log_metric("best_f1", float(best_res["f1"]))
                    mlflow.log_metric("best_accuracy", float(best_res["accuracy"]))

            st.session_state.training_results = pyramid.results
            st.session_state.pyramid = pyramid
            st.session_state.layer_strategy_map = layer_strategy_map
            st.session_state.artifacts_dir = str(artifacts_dir)
            st.session_state.run_training = False
            status_text.text("✅ Treinamento concluído!")
            progress_bar.progress(1.0)

        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.session_state.run_training = False

# Display results
if st.session_state.training_results:
    results = st.session_state.training_results
    layer_strategy_map = st.session_state.layer_strategy_map or {}
    artifacts_dir = st.session_state.artifacts_dir
    if artifacts_dir:
        st.success(f"Resultados e artefatos salvos em: {artifacts_dir}")
    
    # Metrics overview
    st.markdown("<h2 class='sub-header'>📊 Performance Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_f1 = max(result['f1'] for result in results)
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🏆 Best F1</h3>
            <h1 style='color: #28a745;'>{best_f1:.3f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_acc = max(result['accuracy'] for result in results)
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🎯 Best Accuracy</h3>
            <h1 style='color: #007bff;'>{best_acc:.3f}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_models = len(results)
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🤖 Total Models</h3>
            <h1 style='color: #6f42c1;'>{total_models}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_layers = max(result['layer'] for result in results) + 1
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🏗️ Layers Trained</h3>
            <h1 style='color: #fd7e14;'>{total_layers}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced visualization section
    st.markdown("<h2 class='sub-header'>🎨 Enhanced Visualization</h2>", unsafe_allow_html=True)
    
    # Create enhanced visualization
    viz_fig = create_enhanced_ensemble_visualization(
        results, 
        st.session_state.pyramid, 
        show_connections, 
        show_performance,
        layer_strategy_map
    )
    if viz_fig:
        st.plotly_chart(viz_fig, use_container_width=True)
    
    # Interactive performance dashboard
    st.markdown("<h2 class='sub-header'>📈 Performance Analysis</h2>", unsafe_allow_html=True)
    create_interactive_performance_dashboard(results)
    
    # Layer analysis
    st.markdown("<h2 class='sub-header'>🔍 Layer Analysis</h2>", unsafe_allow_html=True)
    create_layer_analysis_panel(results)

else:
    # Welcome and instructions
    st.markdown("""
    ## 🎯 Welcome to the Enhanced Flexible Ensemble Pyramid Interface
    
    This interactive tool provides advanced visualization and analysis capabilities for hierarchical ensemble 
    learning. Experience real-time model connections, performance heatmaps, and detailed layer analysis.
    
    ### 🚀 Enhanced Features
    
    - **Tree Branching Visualization**: See how models connect across layers with performance-based coloring
    - **Performance Heatmaps**: Visual overlay showing model performance across the pyramid
    - **Interactive Dashboard**: Multiple views for comprehensive analysis
    - **Enhanced Styling**: Modern UI with gradient backgrounds and hover effects
    - **Real-time Connections**: Dynamic connections that reflect performance relationships
    
    ### 📊 What You'll Discover
    
    - How different model types perform at each layer
    - Performance trends across the hierarchical structure
    - Optimal model combinations and connections
    - Layer-by-layer performance breakdown
    - Model distribution and specialization patterns
    
    ### ⚡ Quick Start
    
    1. Configure your experiment in the sidebar
    2. Click "Run Ensemble Training"
    3. Explore the interactive visualizations
    4. Analyze performance across different views
    5. Drill down into specific layers for detailed insights
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3>🌳 Tree Connections</h3>
            <p>Visualize model relationships with performance-based branching</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3>🔥 Heatmaps</h3>
            <p>See performance patterns with interactive heatmap overlays</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3>📊 Multi-view</h3>
            <p>Explore data through multiple interactive visualization tabs</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 2rem;'>
    <p style='font-size: 1.1rem;'>Built with ❤️ using Streamlit | Enhanced Flexible Ensemble Pyramid</p>
    <p style='font-size: 0.9rem;'>MLOps Experiments - Advanced Visualization Interface</p>
</div>
""", unsafe_allow_html=True)
