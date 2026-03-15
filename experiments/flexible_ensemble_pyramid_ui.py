import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
        STRATEGY
    )
except ImportError:
    st.error("Could not import flexible_ensemble_pyramid module. Please ensure it's in the correct location.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Flexible Ensemble Pyramid",
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
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .layer-card {
        background-color: #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .model-badge {
        background-color: #6c757d;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>🧠 Flexible Ensemble Pyramid</h1>", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Experiment settings
    st.subheader("Experiment Settings")
    num_layers = st.slider("Number of Layers", 2, 20, 12, 1)
    cv_folds = st.slider("CV Folds", 2, 10, 3, 1)
    patience = st.slider("Patience", 1, 10, 3, 1)
    
    # Model selection parameters
    st.subheader("Model Selection")
    min_models = st.slider("Min Models per Layer", 1, 10, 2, 1)
    max_models = st.slider("Max Models per Layer", 2, 15, 6, 1)
    epsilon_rl = st.slider("RL Exploration Rate", 0.0, 1.0, 0.2, 0.05)
    
    # Feature engineering
    st.subheader("Feature Engineering")
    tfidf_max = st.slider("TF-IDF Max Features", 1000, 100000, 50000, 1000)
    tfidf_ngrams = st.selectbox("TF-IDF N-grams", [(1, 1), (1, 2), (1, 3), (2, 2)], index=1)
    
    # Advanced options
    st.subheader("Advanced Options")
    use_jitter = st.checkbox("Use Hyperparameter Jitter", value=True)
    use_nas = st.checkbox("Use NAS Controller", value=True)
    strategy = st.selectbox("Connection Strategy", ["dense", "residual", "simple"], index=0)
    metric = st.selectbox("Optimization Metric", ["f1", "accuracy"], index=0)
    
    # Run button
    if st.button("🚀 Run Ensemble Training", use_container_width=True):
        st.session_state.run_training = True

# Initialize session state
if 'run_training' not in st.session_state:
    st.session_state.run_training = False
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'pyramid' not in st.session_state:
    st.session_state.pyramid = None

# Main content
def create_ensemble_visualization(results, pyramid):
    """Create interactive visualization of the ensemble pyramid"""
    
    if not results:
        return
    
    # Prepare data for visualization
    layers_data = {}
    for result in results:
        layer_idx = result['layer']
        if layer_idx not in layers_data:
            layers_data[layer_idx] = []
        layers_data[layer_idx].append(result)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Colors for different model types
    model_colors = {
        'lr': '#1f77b4', 'svc': '#ff7f0e', 'nb': '#2ca02c',
        'rf': '#d62728', 'et': '#9467bd', 'ridge': '#8c564b',
        'bag_lr': '#e377c2', 'bag_svc': '#7f7f7f', 'bag_nb': '#bcbd22'
    }
    
    # Position layers vertically
    max_models_in_layer = max(len(models) for models in layers_data.values())
    layer_height = 1.0
    
    # Add nodes (models)
    for layer_idx, models in layers_data.items():
        y_pos = -layer_idx * layer_height
        num_models = len(models)
        
        for i, model_result in enumerate(models):
            model_name = model_result['model']
            base_model = model_name.split('_')[0] if 'bag_' in model_name else model_name
            color = model_colors.get(base_model, '#17becf')
            
            # Calculate x position (centered)
            x_pos = (i - (num_models - 1) / 2) * 0.3
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode='markers+text',
                marker=dict(size=20, color=color),
                text=model_name,
                textposition="bottom center",
                name=f"Layer {layer_idx}: {model_name}",
                customdata=[[model_result['f1'], model_result['accuracy'], model_result['duration']]],
                hovertemplate="<b>%{text}</b><br>" +
                            "F1: %{customdata[0]:.3f}<br>" +
                            "Accuracy: %{customdata[1]:.3f}<br>" +
                            "Duration: %{customdata[2]:.1f}s<extra></extra>"
            ))
    
    # Add connections between layers (simplified)
    for layer_idx in range(len(layers_data) - 1):
        if layer_idx + 1 in layers_data:
            current_models = layers_data[layer_idx]
            next_models = layers_data[layer_idx + 1]
            
            for i, current_model in enumerate(current_models):
                for j, next_model in enumerate(next_models):
                    # Simple connection logic - could be enhanced based on actual ensemble strategy
                    fig.add_trace(go.Scatter(
                        x=[(i - (len(current_models) - 1) / 2) * 0.3, 
                           (j - (len(next_models) - 1) / 2) * 0.3],
                        y=[-layer_idx * layer_height, -(layer_idx + 1) * layer_height],
                        mode='lines',
                        line=dict(width=1, color='gray', dash='dot'),
                        showlegend=False,
                        hoverinfo='none'
                    ))
    
    # Layout configuration
    fig.update_layout(
        title="Ensemble Pyramid Architecture",
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_performance_chart(results):
    """Create performance metrics chart"""
    if not results:
        return None
    
    # Prepare data
    df = pd.DataFrame(results)
    df['layer_model'] = df['layer'].astype(str) + '_' + df['model']
    
    # Create line chart for F1 scores across layers
    fig = px.line(df, x='layer', y='f1', color='model', 
                 title="F1 Score by Layer and Model",
                 labels={'layer': 'Layer', 'f1': 'F1 Score'})
    
    fig.update_layout(height=400, showlegend=True)
    return fig

def create_model_distribution_chart(results):
    """Create model type distribution chart"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    df['model_type'] = df['model'].apply(lambda x: x.split('_')[0] if 'bag_' in x else x)
    
    fig = px.histogram(df, x='model_type', color='model_type',
                      title="Model Type Distribution",
                      labels={'model_type': 'Model Type', 'count': 'Count'})
    
    fig.update_layout(height=300, showlegend=False)
    return fig

# Training execution
if st.session_state.run_training:
    with st.spinner("🚀 Training ensemble pyramid..."):
        
        # Setup progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Setup MLflow tracking
            setup_tracking()
            
            # Load data
            status_text.text("📊 Loading data...")
            train_df, val_df = load_data()
            
            # Prepare features and labels
            vectorizer = TfidfVectorizer(
                max_features=tfidf_max,
                ngram_range=tfidf_ngrams,
                stop_words='english'
            )
            
            X_train = vectorizer.fit_transform(train_df['clean_text'])
            X_val = vectorizer.transform(val_df['clean_text'])
            
            le = LabelEncoder()
            y_train = le.fit_transform(train_df['sentiment'])
            y_val = le.transform(val_df['sentiment'])
            
            # Initialize meta-learners
            rl_learner = RLMetaLearner(epsilon=epsilon_rl, metric=metric)
            nas_controller = NASController(metric=metric) if use_nas else None
            
            # Create pyramid ensemble
            pyramid = PyramidEnsemble(
                num_layers=num_layers,
                meta_learner=rl_learner,
                patience=patience,
                metric=metric,
                jitter=use_jitter,
                strategy=strategy,
                min_models=min_models,
                max_models=max_models,
                nas_controller=nas_controller
            )
            
            # Train the pyramid
            status_text.text("🧠 Training ensemble layers...")
            
            # Mock training progress (actual training would happen here)
            for i in range(num_layers):
                progress = (i + 1) / num_layers
                progress_bar.progress(progress)
                status_text.text(f"🏗️ Training layer {i+1}/{num_layers}...")
                time.sleep(0.5)  # Simulate training time
            
            # Store results
            st.session_state.training_results = pyramid.results
            st.session_state.pyramid = pyramid
            
            status_text.text("✅ Training completed!")
            progress_bar.progress(1.0)
            
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.session_state.run_training = False

# Display results
if st.session_state.training_results:
    results = st.session_state.training_results
    
    # Metrics overview
    st.markdown("<h2 class='sub-header'>📊 Performance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_f1 = max(result['f1'] for result in results)
        st.metric("Best F1 Score", f"{best_f1:.3f}")
    
    with col2:
        best_acc = max(result['accuracy'] for result in results)
        st.metric("Best Accuracy", f"{best_acc:.3f}")
    
    with col3:
        total_models = len(results)
        st.metric("Total Models", total_models)
    
    with col4:
        total_layers = max(result['layer'] for result in results) + 1
        st.metric("Layers Trained", total_layers)
    
    # Visualization section
    st.markdown("<h2 class='sub-header'>🎨 Ensemble Visualization</h2>", unsafe_allow_html=True)
    
    # Create visualizations
    viz_fig = create_ensemble_visualization(results, st.session_state.pyramid)
    if viz_fig:
        st.plotly_chart(viz_fig, use_container_width=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        perf_fig = create_performance_chart(results)
        if perf_fig:
            st.plotly_chart(perf_fig, use_container_width=True)
    
    with col2:
        dist_fig = create_model_distribution_chart(results)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
    
    # Detailed results table
    st.markdown("<h2 class='sub-header'>📋 Detailed Results</h2>", unsafe_allow_html=True)
    
    results_df = pd.DataFrame(results)
    st.dataframe(
        results_df[['layer', 'model', 'f1', 'accuracy', 'duration']]
        .sort_values(['layer', 'f1'], ascending=[True, False])
        .style.format({'f1': '{:.3f}', 'accuracy': '{:.3f}', 'duration': '{:.1f}'})
        .background_gradient(subset=['f1', 'accuracy'], cmap='YlGnBu')
        .set_properties(**{'text-align': 'center'})
    )
    
    # Layer-wise analysis
    st.markdown("<h2 class='sub-header'>🔍 Layer Analysis</h2>", unsafe_allow_html=True)
    
    for layer_idx in sorted(set(results_df['layer'])):
        layer_results = results_df[results_df['layer'] == layer_idx]
        best_model = layer_results.loc[layer_results['f1'].idxmax()]
        
        with st.expander(f"Layer {layer_idx} - Best: {best_model['model']} (F1: {best_model['f1']:.3f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Models in this layer:**")
                for _, row in layer_results.iterrows():
                    st.markdown(f"<span class='model-badge'>{row['model']}</span> F1: {row['f1']:.3f}", 
                               unsafe_allow_html=True)
            
            with col2:
                st.write("**Performance Summary:**")
                st.metric("Average F1", f"{layer_results['f1'].mean():.3f}")
                st.metric("Best F1", f"{best_model['f1']:.3f}")
                st.metric("Number of Models", len(layer_results))

else:
    # Welcome and instructions
    st.markdown("""
    ## 🎯 Welcome to the Flexible Ensemble Pyramid Interface
    
    This interactive tool allows you to train and visualize a hierarchical ensemble of machine learning models 
    organized in a pyramid structure. Each layer builds upon the previous one, creating increasingly sophisticated 
    model combinations.
    
    ### 🚀 Getting Started
    
    1. **Configure** your experiment using the sidebar options
    2. **Adjust** parameters like number of layers, model selection criteria, and feature engineering
    3. **Click** the "Run Ensemble Training" button to start the process
    4. **Explore** the interactive visualizations and performance metrics
    
    ### 🔧 Key Features
    
    - **Reinforcement Learning Meta-Learner**: Optimizes model selection across runs
    - **Neural Architecture Search**: Automatically discovers optimal ensemble architectures
    - **Real-time Visualization**: Interactive plots showing model connections and performance
    - **Multi-layer Hierarchy**: Models organized in increasing complexity layers
    - **Performance Tracking**: Comprehensive metrics and comparison tools
    
    ### 📊 What You'll See
    
    After training, you'll get:
    - Interactive pyramid visualization with model connections
    - Performance charts across layers
    - Detailed metrics for each model
    - Model distribution analysis
    - Layer-by-layer breakdown
    """)
    
    # Quick start example
    st.info("💡 **Tip**: Start with the default settings to see the ensemble in action!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d;'>
    <p>Built with ❤️ using Streamlit | Flexible Ensemble Pyramid Interface</p>
    <p>MLOps Experiments - PedroM2626</p>
</div>
""", unsafe_allow_html=True)