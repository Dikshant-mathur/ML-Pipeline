# ============================================
# STREAMLIT WEB APPLICATION
# 6 Models, 6 Metrics, Confusion Matrix
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="ML Pipeline Playground",
    page_icon="🤖",
    layout="wide"
)

# ============ TITLE ============
st.title("ML Pipeline Playground")
st.markdown("#### **BITS Pilani - M.Tech DSE**")
st.markdown("---")

# ============ SIDEBAR ============
st.sidebar.image("https://www.bits-pilani.ac.in/UploadedImages/Homepage/logo.png", width=200)
st.sidebar.title("Controls")

# ============ LOAD MODELS ============
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'K-Nearest Neighbor': 'model/k-nearest_neighbor.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    
    # Load scaler
    scaler = joblib.load('model/scaler.pkl') if os.path.exists('model/scaler.pkl') else None
    
    return models, scaler

models_dict, scaler = load_models()

# ============ DATASET UPLOAD ============
st.sidebar.header("1. Upload Test Data (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (test data only)",
    type=['csv']
)

# ============ MODEL SELECTION ============
st.sidebar.header("2. Select Model")
model_names = list(models_dict.keys())
selected_model = st.sidebar.selectbox(
    "Choose Model for Evaluation",
    model_names
)

# ============ MAIN CONTENT ============
if uploaded_file is not None:
    # Load test data
    test_df = pd.read_csv(uploaded_file)
    
    # Assume last column is target
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    st.success(f"Test data loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # Data preview
    with st.expander("Test Data Preview"):
        st.dataframe(test_df.head())
    
    # Model evaluation
    if st.sidebar.button("Run Evaluation", type="primary"):
        st.markdown("---")
        st.header(f"Evaluation Results: {selected_model}")
        
        model = models_dict[selected_model]
        
        # Scaling required?
        if selected_model in ['Logistic Regression', 'K-Nearest Neighbor'] and scaler:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        accuracy = accuracy_score(y_test, y_pred)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        
        try:
            auc = roc_auc_score(y_test, y_proba)
            col2.metric("AUC Score", f"{auc:.4f}")
        except:
            col2.metric("AUC Score", "N/A")
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        col3.metric("Precision", f"{precision:.4f}")
        
        recall = recall_score(y_test, y_pred, zero_division=0)
        col4.metric("Recall", f"{recall:.4f}")
        
        f1 = f1_score(y_test, y_pred, zero_division=0)
        col5.metric("F1 Score", f"{f1:.4f}")
        
        mcc = matthews_corrcoef(y_test, y_pred)
        col6.metric("MCC", f"{mcc:.4f}")
        
        # Confusion Matrix
        st.markdown("---")
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        ax.set_title(f'Confusion Matrix - {selected_model}')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
else:
    # No file uploaded - show info
    st.info("Please upload test dataset from sidebar to begin evaluation")
    
    # Show sample format
    with st.expander("Sample Data Format"):
        st.write("Your CSV should have:")
        st.code("""
- Last column: Target variable (0/1 for binary)
- Other columns: Features (minimum 12 features)
- No header? Check 'Has header' option
        """)

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Developed for ML Assignment 2 | BITS Pilani WILP | Deployed on Streamlit Cloud
</div>
""", unsafe_allow_html=True)