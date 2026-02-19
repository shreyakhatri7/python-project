import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import streamlit.components.v1 as components
warnings.filterwarnings('ignore')

# Import preprocessing module (Member 2)
from preprocessing import DataPreprocessor, preprocess_data, get_data_summary

# Configure page
st.set_page_config(page_title="Fraud Detection Dashboard - Members 1 & 2", layout="wide")

# Title and description
st.title("🔍 Fraud Detection Dashboard")
st.markdown("""
This dashboard provides data analysis (Member 1) and preprocessing (Member 2).
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "Data Upload & Analysis",
    "Data Preprocessing"
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

# Function to load default dataset
def load_default_data():
    try:
        df = pd.read_csv('final_file.csv')
        return df
    except FileNotFoundError:
        return None

# ============================================================================
# PAGE 1: Data Upload & Analysis (Member 1 - EDA)
# ============================================================================
if page == "Data Upload & Analysis":
    st.header("📊 Data Upload & Analysis (Member 1 - EDA)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        
    with col2:
        # Option to use default dataset
        if st.button("Use Default Dataset"):
            default_data = load_default_data()
            if default_data is not None:
                st.session_state.df = default_data
                st.success("✅ Default dataset loaded!")
            else:
                st.error("❌ Default dataset not found!")
    
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    
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
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information
        if st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Missing values visualization
        if df.isnull().sum().sum() > 0:
            st.subheader("Missing Values")
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        labels={'x': 'Columns', 'y': 'Missing Count'},
                        title='Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        if st.expander("📊 Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Target column selection and visualization
        st.subheader("Target Variable Analysis")
        target_cols = df.columns.tolist()
        target_col = st.selectbox("Select target column", target_cols)
        
        if target_col:
            target_counts = df[target_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig = px.pie(values=target_counts.values, 
                           names=target_counts.index,
                           title=f'Distribution of {target_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart
                fig = px.bar(x=target_counts.index, y=target_counts.values,
                           labels={'x': target_col, 'y': 'Count'},
                           title=f'Count by {target_col}')
                st.plotly_chart(fig, use_container_width=True)
            
            # Class balance info
            if len(target_counts) == 2:
                majority_class = target_counts.max()
                minority_class = target_counts.min()
                imbalance_ratio = majority_class / minority_class
                
                st.info(f"""
                **Class Balance:**
                - Majority class: {majority_class} samples
                - Minority class: {minority_class} samples
                - Imbalance ratio: {imbalance_ratio:.2f}:1
                """)
                
                if imbalance_ratio > 3:
                    st.warning("⚠️ Significant class imbalance detected! Consider using SMOTE in preprocessing.")
        
        # Display EDA Report (Member 1's output)
        st.subheader("📈 Full EDA Report (Member 1)")
        st.markdown("This report was generated using ydata-profiling in data.ipynb")
        
        if st.button("Display EDA Report"):
            try:
                with open('eda_report.html', 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    components.html(html_content, height=800, scrolling=True)
            except FileNotFoundError:
                st.error("❌ eda_report.html not found! Please run data.ipynb first to generate the report.")
        
        # Download options
        st.subheader("💾 Download Data")
        st.download_button(
            label="📥 Download Current Dataset",
            data=df.to_csv(index=False),
            file_name="dataset.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Please upload a CSV file or use the default dataset to begin.")

# ============================================================================
# PAGE 2: Data Preprocessing (Member 2)
# ============================================================================
elif page == "Data Preprocessing":
    st.header("🔧 Data Preprocessing (Member 2)")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first on the 'Data Upload & Analysis' page!")
        st.stop()
    
    df = st.session_state.df
    
    st.subheader("Preprocessing Configuration")
    
    # Select target column
    target_col = st.selectbox("Select target column", df.columns.tolist())
    
    # Preprocessing options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Scaling Method**")
        scaling_method = st.radio("Choose scaling method", 
                                 ['standard', 'minmax'],
                                 help="Standard: mean=0, std=1 | MinMax: scale to [0,1]")
        
        st.markdown("**Encoding Method**")
        encoding_method = st.radio("Choose encoding method",
                                  ['onehot', 'label'],
                                  help="OneHot: create binary columns | Label: assign numbers")
    
    with col2:
        st.markdown("**Class Imbalance Handling**")
        imbalance_method = st.selectbox("Choose imbalance handling",
                                       ['none', 'smote', 'undersample', 'smotetomek'],
                                       help="SMOTE: oversample minority | Undersample: reduce majority")
        
        st.markdown("**Test Size**")
        test_size = st.slider("Test set proportion", 0.1, 0.4, 0.2, 0.05)
    
    st.markdown("**Feature Engineering**")
    generate_features = st.checkbox("Generate additional features", value=False,
                                   help="Create rolling averages and transaction ratios")
    
    # Preprocessing summary
    if st.expander("📋 Preprocessing Summary"):
        st.markdown(f"""
        **Configuration:**
        - Target: `{target_col}`
        - Scaling: `{scaling_method}`
        - Encoding: `{encoding_method}`
        - Imbalance handling: `{imbalance_method}`
        - Test size: `{test_size*100}%`
        - Feature engineering: `{generate_features}`
        """)
    
    # Run preprocessing
    if st.button("🚀 Run Preprocessing", type="primary"):
        if target_col:
            with st.spinner("Processing data..."):
                try:
                    # Convert 'none' to None for imbalance_method
                    imbalance_param = None if imbalance_method == 'none' else imbalance_method
                    
                    # Run preprocessing (Member 2's function)
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
                    st.subheader("Preprocessing Results")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Training Samples", X_train.shape[0])
                    with col2:
                        st.metric("Testing Samples", X_test.shape[0])
                    with col3:
                        st.metric("Total Features", X_train.shape[1])
                    with col4:
                        fraud_count = y_train.sum() if hasattr(y_train, 'sum') else 0
                        st.metric("Fraud in Training", fraud_count)
                    
                    # Training set distribution
                    st.subheader("Training Set Class Distribution")
                    train_counts = pd.Series(y_train).value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(values=train_counts.values, 
                                   names=train_counts.index,
                                   title='Training Set Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(x=train_counts.index, y=train_counts.values,
                                   labels={'x': 'Class', 'y': 'Count'},
                                   title='Training Set Counts')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature names
                    if st.expander("📋 Generated Features"):
                        st.info(f"Total features after preprocessing: {len(preprocessor.feature_names)}")
                        feature_df = pd.DataFrame({
                            'Index': range(len(preprocessor.feature_names)),
                            'Feature Name': preprocessor.feature_names
                        })
                        st.dataframe(feature_df, use_container_width=True)
                    
                    # Preview preprocessed data
                    if st.expander("👁️ Preview Preprocessed Data"):
                        st.markdown("**Training Data (first 10 rows):**")
                        st.dataframe(X_train.head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {str(e)}")
                    st.exception(e)
        else:
            st.warning("⚠️ Please select a target column!")
    
    # Download preprocessed data
    if st.session_state.X_train is not None:
        st.subheader("💾 Download Preprocessed Data")
        
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
                file_name="preprocessed_train_data.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📥 Download Test Data",
                data=test_data.to_csv(index=False),
                file_name="preprocessed_test_data.csv",
                mime="text/csv"
            )

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Dashboard Info")
st.sidebar.info("""
**Fraud Detection Dashboard**
*Members 1 & 2*

**Member 1 - EDA:**
- Data exploration
- Statistical analysis
- EDA report display

**Member 2 - Preprocessing:**
- Data cleaning
- Feature engineering
- Scaling & encoding
- Class imbalance handling

*Member 3 (ML Models) will be added later*
""")
