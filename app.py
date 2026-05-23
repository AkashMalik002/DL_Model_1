"""
Interactive Streamlit Web Application for Deep Learning Models
Demonstrates 4 ML models trained across 5 real-world domains
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import custom modules
from datasets import create_domain_datasets, display_dataset_summary
from models import LogisticRegression, LinearRegression, SoftmaxRegression, MultiLayerPerceptron
from metrics import EvaluationMetrics
from preprocessing import preprocess_data
from training import train_all_domains
from analysis import get_domain_recommendations

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Deep Learning Model Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Home", "📊 Dataset Explorer", "🤖 Model Training", "📈 Performance Analysis", "🔮 Make Predictions"]
)

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = create_domain_datasets()
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# ==================== PAGE: HOME ====================
if page == "🏠 Home":
    st.title("🤖 Deep Learning Model Demonstration Platform")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## 📋 Project Overview
        
        This application demonstrates a complete **Deep Learning pipeline** with:
        
        - **4 ML Models** implemented from scratch using NumPy:
          - Logistic Regression
          - Linear Regression
          - Softmax Regression
          - Multi-Layer Perceptron (MLP)
        
        - **5 Real-World Domains**:
          - 📝 NLP (Review Classification)
          - 🌾 Agriculture (Crop Health)
          - 💾 Database (Infrastructure Monitoring)
          - 🏥 Healthcare (Patient Readmission)
          - 💰 FinTech (Fraud Detection)
        """)
    
    with col2:
        st.markdown("""
        ## ✨ Key Features
        
        ✅ **Interactive Data Exploration** - Explore dataset statistics  
        ✅ **Real-time Model Training** - Train models on selected domains  
        ✅ **Performance Metrics** - Compare accuracy, precision, recall, F1-score  
        ✅ **Visualization Dashboard** - Loss curves and metrics comparison  
        ✅ **Predictions** - Make predictions on custom data  
        ✅ **Domain Analysis** - Insights on model behavior per domain  
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Quick Start
    
    1. **Dataset Explorer** - View datasets from all domains
    2. **Model Training** - Train models and monitor training progress
    3. **Performance Analysis** - Compare model performance across domains
    4. **Make Predictions** - Test models with custom inputs
    """)

# ==================== PAGE: DATASET EXPLORER ====================
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown("---")
    
    # Select domain
    domain_mapping = {
        'nlp': 'Domain 1: NLP Review Classification',
        'agriculture': 'Domain 2: Agriculture Crop Health',
        'database': 'Domain 3: Database Infrastructure Monitoring',
        'healthcare': 'Domain 4: Healthcare Patient Readmission',
        'fintech': 'Domain 5: FinTech Fraud Detection'
    }
    
    selected_domain_name = st.selectbox("Select a Domain:", list(domain_mapping.values()))
    selected_domain = [k for k, v in domain_mapping.items() if v == selected_domain_name][0]
    
    dataset_info = st.session_state.datasets[selected_domain]
    df = dataset_info['data']
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Classes", len(df['target'].unique()))
    with col4:
        st.metric("Class Balance", f"{(df['target'].sum() / len(df) * 100):.1f}% Positive")
    
    st.markdown("---")
    
    # Display dataset statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Class distribution
    st.subheader("Class Distribution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    class_counts = df['target'].value_counts()
    ax1.bar(["Negative (0)", "Positive (1)"], class_counts.values, color=['#FF6B6B', '#4ECDC4'])
    ax1.set_title("Sample Count by Class")
    ax1.set_ylabel("Count")
    
    ax2.pie(class_counts.values, labels=["Negative (0)", "Positive (1)"], 
            autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
    ax2.set_title("Class Distribution (%)")
    
    st.pyplot(fig, use_container_width=True)

# ==================== PAGE: MODEL TRAINING ====================
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Select Domain & Training Parameters**")
    with col2:
        train_button = st.button("▶️ Train Models", key="train_btn", use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    domain_mapping = {
        'nlp': 'Domain 1: NLP Review Classification',
        'agriculture': 'Domain 2: Agriculture Crop Health',
        'database': 'Domain 3: Database Infrastructure Monitoring',
        'healthcare': 'Domain 4: Healthcare Patient Readmission',
        'fintech': 'Domain 5: FinTech Fraud Detection'
    }
    
    with col1:
        selected_domain_name = st.selectbox("Select Domain:", list(domain_mapping.values()))
        selected_domain = [k for k, v in domain_mapping.items() if v == selected_domain_name][0]
    
    with col2:
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    
    with col3:
        epochs = st.slider("Epochs", 50, 300, 150, step=10)
    
    st.markdown("---")
    
    if train_button:
        with st.spinner("🔄 Training models... This may take a moment."):
            dataset_info = st.session_state.datasets[selected_domain]
            df = dataset_info['data']
            
            # Preprocess
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
            
            # Train Logistic Regression
            lr_model = LogisticRegression(learning_rate=learning_rate, epochs=epochs)
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict(X_test)
            
            # Train MLP
            mlp_model = MultiLayerPerceptron(
                hidden_layers=[64, 32],
                learning_rate=learning_rate,
                epochs=epochs,
                task='binary',
                num_classes=2
            )
            mlp_model.fit(X_train, y_train)
            mlp_pred = mlp_model.predict(X_test)
            
            # Calculate metrics
            lr_metrics = {
                'Accuracy': EvaluationMetrics.accuracy(y_test, lr_pred),
                'Precision': EvaluationMetrics.precision(y_test, lr_pred),
                'Recall': EvaluationMetrics.recall(y_test, lr_pred),
                'F1-Score': EvaluationMetrics.f1_score(y_test, lr_pred),
            }
            
            mlp_metrics = {
                'Accuracy': EvaluationMetrics.accuracy(y_test, mlp_pred),
                'Precision': EvaluationMetrics.precision(y_test, mlp_pred),
                'Recall': EvaluationMetrics.recall(y_test, mlp_pred),
                'F1-Score': EvaluationMetrics.f1_score(y_test, mlp_pred),
            }
            
            st.session_state.training_results = {
                'domain': selected_domain,
                'lr_model': lr_model,
                'mlp_model': mlp_model,
                'lr_metrics': lr_metrics,
                'mlp_metrics': mlp_metrics,
                'lr_pred': lr_pred,
                'mlp_pred': mlp_pred,
                'y_test': y_test,
                'X_test': X_test,
                'scaler': scaler,
                'X_train': X_train,
            }
        
        st.success("✅ Training Complete!")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Logistic Regression Metrics")
            metrics_df = pd.DataFrame(list(lr_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 MLP Metrics")
            metrics_df = pd.DataFrame(list(mlp_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(metrics_df, use_container_width=True)
        
        # Loss history visualization
        st.markdown("---")
        st.subheader("📈 Training Loss Over Epochs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(lr_model.loss_history, linewidth=2, label='Loss', color='#4ECDC4')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Logistic Regression Training Loss', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(mlp_model.loss_history, linewidth=2, label='Loss', color='#FF6B6B')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('MLP Training Loss', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("👆 Click 'Train Models' to start training!")

# ==================== PAGE: PERFORMANCE ANALYSIS ====================
elif page == "📈 Performance Analysis":
    st.title("📈 Performance Analysis")
    st.markdown("---")
    
    if st.session_state.training_results is None:
        st.warning("⚠️ Please train models first in the 'Model Training' tab!")
    else:
        results = st.session_state.training_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Logistic Regression Performance")
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics = list(results['lr_metrics'].keys())
            values = list(results['lr_metrics'].values())
            bars = ax.barh(metrics, values, color='#4ECDC4')
            ax.set_xlim(0, 1)
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + 0.02, i, f'{value:.3f}', va='center', fontweight='bold')
            ax.set_xlabel('Score', fontsize=11)
            ax.set_title('Logistic Regression Metrics', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.subheader("MLP Performance")
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics = list(results['mlp_metrics'].keys())
            values = list(results['mlp_metrics'].values())
            bars = ax.barh(metrics, values, color='#FF6B6B')
            ax.set_xlim(0, 1)
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + 0.02, i, f'{value:.3f}', va='center', fontweight='bold')
            ax.set_xlabel('Score', fontsize=11)
            ax.set_title('MLP Metrics', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Metrics Comparison")
        
        comparison_df = pd.DataFrame({
            'Metric': results['lr_metrics'].keys(),
            'Logistic Regression': results['lr_metrics'].values(),
            'MLP': results['mlp_metrics'].values()
        })
        
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(comparison_df))
        width = 0.35
        ax.bar(x - width/2, comparison_df['Logistic Regression'], width, label='Logistic Regression', color='#4ECDC4')
        ax.bar(x + width/2, comparison_df['MLP'], width, label='MLP', color='#FF6B6B')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Metric'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Detailed Metrics Table")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ==================== PAGE: MAKE PREDICTIONS ====================
elif page == "🔮 Make Predictions":
    st.title("🔮 Make Predictions")
    st.markdown("---")
    
    if st.session_state.training_results is None:
        st.warning("⚠️ Please train models first in the 'Model Training' tab!")
    else:
        results = st.session_state.training_results
        
        st.markdown("### Enter Feature Values")
        st.info("📝 Enter values for all features (the scaler will normalize them automatically)")
        
        num_features = results['X_train'].shape[1]
        input_values = []
        
        # Create input fields in columns for better layout
        cols = st.columns(4)
        for i in range(num_features):
            with cols[i % 4]:
                value = st.number_input(
                    f"Feature {i+1}",
                    value=0.0,
                    step=0.1,
                    format="%.2f"
                )
                input_values.append(value)
        
        st.markdown("---")
        
        if st.button("🔮 Make Prediction", use_container_width=True):
            # Prepare input
            X_input = np.array(input_values).reshape(1, -1)
            X_input_scaled = results['scaler'].transform(X_input)
            
            # Get predictions
            lr_pred = results['lr_model'].predict(X_input_scaled)[0]
            mlp_pred = results['mlp_model'].predict(X_input_scaled)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Logistic Regression Prediction")
                pred_class = "✅ Positive (Class 1)" if lr_pred >= 0.5 else "❌ Negative (Class 0)"
                st.markdown(f"### {pred_class}")
                st.metric("Confidence", f"{max(lr_pred, 1-lr_pred):.2%}")
            
            with col2:
                st.subheader("MLP Prediction")
                pred_class = "✅ Positive (Class 1)" if mlp_pred >= 0.5 else "❌ Negative (Class 0)"
                st.markdown(f"### {pred_class}")
                st.metric("Confidence", f"{max(mlp_pred, 1-mlp_pred):.2%}")
            
            st.markdown("---")
            st.subheader("📊 Prediction Probabilities")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            models = ['Logistic Regression', 'MLP']
            positive_probs = [lr_pred, mlp_pred]
            negative_probs = [1 - lr_pred, 1 - mlp_pred]
            
            x = np.arange(len(models))
            width = 0.35
            ax.bar(x, negative_probs, width, label='Class 0 (Negative)', color='#FF6B6B')
            ax.bar(x, positive_probs, width, bottom=negative_probs, label='Class 1 (Positive)', color='#4ECDC4')
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title('Prediction Probabilities by Model', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.set_ylim(0, 1)
            st.pyplot(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p><strong>Deep Learning Assignment Demonstration Platform</strong></p>
        <p>🚀 Powered by Streamlit | 🤖 Built with NumPy | 📊 Models from Scratch</p>
        <p style="font-size: 0.9em;">Student ID: adid1212 | 2026</p>
    </div>
""", unsafe_allow_html=True)
