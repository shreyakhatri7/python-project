import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import minmax_scale
import xgboost as xgb
import joblib
import os

# Import your preprocessing module]
from preprocessing import DataPreprocessor, preprocess_data, get_data_summary

# Configure page
st.set_page_config(page_title="Advanced Fraud Detection Dashboard", layout="wide")

# Title and description
st.title("🔍 Advanced Fraud Detection Dashboard")
st.markdown("""
This dashboard provides comprehensive fraud detection analysis with automated preprocessing, 
multiple ML models, and interactive predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "Data Upload & Analysis",
    "Data Preprocessing", 
    "Model Training & Evaluation",
    "Predictions",
    "Feature Analysis"
])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'models' not in st.session_state:
    st.session_state.models = {}

# Function to load default dataset
def load_default_data():
    try:
        df = pd.read_csv('final_file.csv')
        return df
    except FileNotFoundError:
        return None

# Page 1: Data Upload & Analysis
if page == "Data Upload & Analysis":
    st.header("📊 Data Upload & Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        
        # Option to use default dataset
        if st.button("Use Default Dataset (final_file.csv)"):
            default_data = load_default_data()
            if default_data is not None:
                st.session_state.df = default_data
                st.success("Default dataset loaded successfully!")
            else:
                st.error("Default dataset not found!")
    
    with col2:
        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
    
    # Display data if available
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Display data
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Target column selection
        st.subheader("Target Column Selection")
        target_col = st.selectbox("Select the target column", df.columns, key="target_select")
        
        if target_col:
            # Display target distribution
            target_counts = df[target_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Target Distribution")
                fig = px.pie(values=target_counts.values, names=target_counts.index, 
                           title=f"Distribution of {target_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Class Balance")
                imbalance_ratio = target_counts.min() / target_counts.max()
                st.metric("Imbalance Ratio", f"{imbalance_ratio:.4f}")
                
                if imbalance_ratio < 0.1:
                    st.warning("⚠️ Highly imbalanced dataset detected!")
                elif imbalance_ratio < 0.3:
                    st.info("ℹ️ Moderately imbalanced dataset")
                else:
                    st.success("✅ Relatively balanced dataset")
        
        # Data summary statistics
        if st.expander("📈 Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Missing values analysis
        if st.expander("🔍 Missing Values Analysis"):
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing_Count': df.isnull().sum().values,
                'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                fig = px.bar(missing_df, x='Column', y='Missing_Percentage', 
                           title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")

            # Display member1 EDA report if available
            if os.path.exists('eda_report.html'):
                if st.button("🔎 Show EDA Report (Member 1)"):
                    with open('eda_report.html', 'r', encoding='utf-8') as f:
                        html = f.read()
                    st.components.v1.html(html, height=800, scrolling=True)

# Page 2: Data Preprocessing
elif page == "Data Preprocessing":
    st.header("⚙️ Data Preprocessing")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        st.stop()
    
    df = st.session_state.df
    
    # Preprocessing configuration
    st.subheader("Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_col = st.selectbox("Select Target Column", df.columns, key="preprocess_target")
        scaling_method = st.selectbox("Scaling Method", ["standard", "minmax"])
        encoding_method = st.selectbox("Encoding Method", ["onehot", "label"])
    
    with col2:
        imbalance_method = st.selectbox("Imbalance Handling", ["smote", "undersample", "smotetomek", "none"])
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        generate_features = st.checkbox("Generate New Features", value=True)
    
    if st.button("🚀 Run Preprocessing", type="primary"):
        if target_col:
            with st.spinner("Processing data..."):
                try:
                    # Convert 'none' to None for imbalance_method
                    imbalance_param = None if imbalance_method == 'none' else imbalance_method
                    
                    # Run preprocessing
                    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
                        df=df,
                        target=target_col,
                        scaling_method=scaling_method,
                        encoding_method=encoding_method,
                        imbalance_method=imbalance_param,
                        test_size=test_size,
                        generate_new_features=generate_features
                    )
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.preprocessor = preprocessor
                    st.session_state.target_col = target_col
                    
                    st.success("✅ Preprocessing completed successfully!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Training Samples", X_train.shape[0])
                    with col2:
                        st.metric("Testing Samples", X_test.shape[0])
                    with col3:
                        st.metric("Total Features", X_train.shape[1])
                    with col4:
                        st.metric("Class Distribution", f"{y_train.value_counts().to_dict()}")
                    
                    # Feature names
                    if st.expander("📋 Generated Features"):
                        feature_df = pd.DataFrame({
                            'Feature_Name': preprocessor.feature_names,
                            'Index': range(len(preprocessor.feature_names))
                        })
                        st.dataframe(feature_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {str(e)}")
        else:
            st.warning("⚠️ Please select a target column!")

# Page 3: Model Training & Evaluation  
elif page == "Model Training & Evaluation":
    st.header("🤖 Model Training & Evaluation")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Please run preprocessing first!")
        st.stop()
    
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    
    # Model selection
    st.subheader("Model Selection")
    selected_models = st.multiselect("Choose models to train", [
        "Random Forest", "Logistic Regression", "Gradient Boosting", "SVM", "XGBoost", "Isolation Forest"
    ], default=["Random Forest", "Logistic Regression"])
    
    if st.button("🏋️ Train Models", type="primary"):
        if selected_models:
            progress_bar = st.progress(0)
            results = {}
            
            models_dict = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                "Isolation Forest": IsolationForest(contamination='auto', random_state=42)
            }
            
            for i, model_name in enumerate(selected_models):
                with st.spinner(f"Training {model_name}..."):
                    model = models_dict[model_name]

                    # Isolation Forest (unsupervised anomaly detection)
                    if model_name == 'Isolation Forest':
                        model.fit(X_train)

                        # predict: -1 for anomaly, 1 for normal
                        preds = model.predict(X_test)
                        # Map anomalies (-1) to 1 (fraud), normal (1) to 0
                        y_pred = np.where(preds == -1, 1, 0)

                        # Anomaly scores (higher -> more abnormal); invert so higher means more fraudlike
                        scores = -model.decision_function(X_test)
                        y_pred_proba = minmax_scale(scores)

                        # Compute metrics if binary target
                        if len(np.unique(y_test)) == 2:
                            auc_val = roc_auc_score(y_test, y_pred_proba)
                        else:
                            auc_val = 0

                    else:
                        # Supervised models
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        if hasattr(model, 'predict_proba'):
                            # probability for positive class
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                        elif hasattr(model, 'decision_function'):
                            # fallback to decision_function then scale
                            y_pred_proba = minmax_scale(model.decision_function(X_test))
                        else:
                            y_pred_proba = np.array(y_pred)

                        auc_val = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) == 2 else 0

                    # Metrics
                    results[model_name] = {
                        'model': model,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                        'auc': auc_val,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }

                    # Save model to disk
                    os.makedirs('saved_models', exist_ok=True)
                    model_path = os.path.join('saved_models', f"{model_name.replace(' ', '_')}.joblib")
                    try:
                        joblib.dump(model, model_path)
                        results[model_name]['model_file'] = model_path
                    except Exception as e:
                        results[model_name]['model_file'] = None
                    
                progress_bar.progress((i + 1) / len(selected_models))
            
            st.session_state.models = results
            
            # Display results
            st.subheader("📊 Model Comparison")
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[model]['accuracy'] for model in results.keys()],
                'Precision': [results[model]['precision'] for model in results.keys()],
                'Recall': [results[model]['recall'] for model in results.keys()],
                'F1-Score': [results[model]['f1'] for model in results.keys()],
                'AUC': [results[model]['auc'] for model in results.keys()]
            })
            
            st.dataframe(metrics_df.round(4), use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Metrics comparison chart
                fig = px.bar(metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                           x='Model', y='Score', color='Metric', barmode='group',
                           title="Model Performance Comparison")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ROC curves (if binary classification)
                if len(np.unique(y_test)) == 2:
                    fig = go.Figure()
                    
                    for model_name in results.keys():
                        fpr, tpr, _ = roc_curve(y_test, results[model_name]['probabilities'])
                        auc_score = results[model_name]['auc']
                        
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            name=f"{model_name} (AUC={auc_score:.3f})",
                            mode='lines'
                        ))
                    
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                           mode='lines', name='Random',
                                           line=dict(dash='dash')))
                    
                    fig.update_layout(
                        title="ROC Curves",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrices
            st.subheader("📈 Confusion Matrices")
            
            cols = st.columns(len(selected_models))
            
            for i, model_name in enumerate(selected_models):
                with cols[i]:
                    cm = confusion_matrix(y_test, results[model_name]['predictions'])
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title(f"{model_name}")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    
                    st.pyplot(fig)
                    plt.close()
        else:
            st.warning("⚠️ Please select at least one model!")

# Page 4: Predictions
elif page == "Predictions":
    st.header("🔮 Make Predictions")
    
    # Allow using in-session trained models OR loading saved models
    saved_models = []
    if os.path.exists('saved_models'):
        saved_models = [f for f in os.listdir('saved_models') if f.endswith('.joblib')]

    st.subheader("Model Source")
    source_choice = st.radio("Choose model source", ["In-session models", "Load saved model"]) if (st.session_state.models or saved_models) else st.radio("Model source", ["Load saved model"]) 

    model = None
    selected_model_name = None

    if source_choice == "In-session models":
        if not st.session_state.models:
            st.warning("No in-session models available. Train models or load a saved model.")
            st.stop()
        available_models = list(st.session_state.models.keys())
        selected_model_name = st.selectbox("Choose model for prediction", available_models)
        model = st.session_state.models[selected_model_name]['model']
    else:
        if not saved_models:
            st.warning("No saved models found in saved_models/. Train and save a model first.")
            st.stop()
        selected_file = st.selectbox("Choose saved model file", saved_models)
        if st.button("Load selected model"):
            model_path = os.path.join('saved_models', selected_file)
            try:
                model = joblib.load(model_path)
                st.success(f"Loaded model: {selected_file}")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()

    st.subheader("Input Features")

    if 'preprocessor' not in st.session_state or st.session_state.preprocessor is None:
        st.warning("Please run preprocessing first so the preprocessor and feature list are available.")
        st.stop()

    feature_names = st.session_state.preprocessor.feature_names
    prediction_raw = {}

    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            sample_vals = st.session_state.X_train[feature].unique()[:20]
            if len(sample_vals) <= 2 and set(sample_vals).issubset({0, 1}):
                prediction_raw[feature] = st.selectbox(f"{feature}", [0, 1], key=f"pred_{feature}")
            elif len(sample_vals) <= 20:
                try:
                    opts = sorted([v for v in sample_vals if pd.notna(v)])
                    prediction_raw[feature] = st.selectbox(f"{feature}", opts, key=f"pred_{feature}")
                except Exception:
                    prediction_raw[feature] = st.text_input(f"{feature}", key=f"pred_{feature}")
            else:
                min_val = float(st.session_state.X_train[feature].min())
                max_val = float(st.session_state.X_train[feature].max())
                mean_val = float(st.session_state.X_train[feature].mean())
                prediction_raw[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val, key=f"pred_{feature}")

    if st.button("🎯 Make Prediction", type="primary"):
        input_raw_df = pd.DataFrame([prediction_raw])

        # Transform raw input using preprocessor (if available)
        preprocessor = st.session_state.preprocessor
        try:
            input_df = preprocessor.transform_new_data(input_raw_df)
        except Exception:
            # If transform_new_data fails, try to use columns as-is
            input_df = input_raw_df

        if model is None:
            st.error("No model loaded for prediction.")
            st.stop()

        # Align columns
        missing_cols = [c for c in st.session_state.preprocessor.feature_names if c not in input_df.columns]
        for c in missing_cols:
            input_df[c] = 0
        input_df = input_df[st.session_state.preprocessor.feature_names]

        # Prediction
        try:
            pred = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Get probability / score
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[0]
            positive_proba = proba[1] if len(proba) > 1 else proba[0]
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(input_df)
            positive_proba = float(minmax_scale(score)[0])
        else:
            positive_proba = None

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Result")
            if pred == 1:
                st.error("🚨 **FRAUDULENT TRANSACTION DETECTED**")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")
            st.metric("Predicted Class", pred)

        with col2:
            if positive_proba is not None:
                st.subheader("Fraud Probability")
                st.metric("Fraud Probability", f"{positive_proba:.3f}")
                prob_bar = px.bar(x=[positive_proba, 1 - positive_proba], y=["Fraud", "Legit"], orientation='h')
                st.plotly_chart(prob_bar, use_container_width=True)

# Page 5: Feature Analysis
elif page == "Feature Analysis":
    st.header("📈 Feature Analysis")
    
    if st.session_state.models is None or len(st.session_state.models) == 0:
        st.warning("⚠️ Please train models first!")
        st.stop()
    
    # Select model for feature analysis
    available_models = list(st.session_state.models.keys())
    selected_model = st.selectbox("Choose model for feature analysis", available_models)
    
    model = st.session_state.models[selected_model]['model']
    feature_names = st.session_state.preprocessor.feature_names
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        # Top 20 features
        top_features = importance_df.tail(20)
        
        fig = px.bar(top_features, x='Importance', y='Feature', orientation='h',
                    title=f"Top 20 Most Important Features - {selected_model}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        if st.expander("📋 All Feature Importances"):
            st.dataframe(importance_df.sort_values('Importance', ascending=False), use_container_width=True)
    
    # Correlation analysis
    if st.expander("🔗 Feature Correlations"):
        # Calculate correlation with target
        X_with_target = st.session_state.X_train.copy()
        X_with_target[st.session_state.target_col] = st.session_state.y_train
        
        correlations = X_with_target.corr()[st.session_state.target_col].abs().sort_values(ascending=False)
        
        # Top correlated features
        top_corr = correlations.head(20).drop(st.session_state.target_col)
        
        fig = px.bar(x=top_corr.values, y=top_corr.index, orientation='h',
                    title="Top 20 Features Correlated with Target")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Dashboard Info")
st.sidebar.info("""
**Advanced Fraud Detection Dashboard**

✅ Automated data preprocessing  
✅ Feature engineering  
✅ Multiple ML models  
✅ Interactive predictions  
✅ Feature analysis  

Built with comprehensive preprocessing module
""")

# Additional utility functions
def download_processed_data():
    """Allow users to download processed data"""
    if st.session_state.X_train is not None:
        # Combine training data
        train_data = st.session_state.X_train.copy()
        train_data[st.session_state.target_col] = st.session_state.y_train
        
        # Combine test data  
        test_data = st.session_state.X_test.copy()
        test_data[st.session_state.target_col] = st.session_state.y_test
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Training Data",
                data=train_data.to_csv(index=False),
                file_name="processed_train_data.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📥 Download Test Data", 
                data=test_data.to_csv(index=False),
                file_name="processed_test_data.csv",
                mime="text/csv"
            )

# Add download option to preprocessing page
if page == "Data Preprocessing" and st.session_state.X_train is not None:
    st.subheader("💾 Download Processed Data")
    download_processed_data()