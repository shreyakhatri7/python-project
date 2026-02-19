# 🔍 Advanced Fraud Detection Dashboard

A comprehensive Streamlit dashboard that integrates your sophisticated preprocessing module with machine learning models for fraud detection analysis.

## 🚀 Features

### 📊 Data Upload & Analysis
- **File Upload**: Upload CSV files or use the default dataset (`final_file.csv`)
- **Dataset Overview**: View dataset statistics, shape, and memory usage
- **Target Analysis**: Automatic target distribution analysis and class balance detection
- **Missing Values**: Comprehensive missing values analysis with visualizations

### ⚙️ Data Preprocessing
- **Automated Column Detection**: Automatic identification of numeric, categorical, and ID columns
- **Missing Value Handling**: Smart imputation using median for numeric and mode for categorical features
- **Feature Encoding**: Choice between Label Encoding and One-Hot Encoding
- **Feature Scaling**: StandardScaler or MinMaxScaler options
- **Class Imbalance**: SMOTE, RandomUnderSampler, SMOTETomek, or no balancing
- **Feature Engineering**: Automatic generation of fraud-specific features:
  - Transaction amount transformations (log, percentiles)
  - Time-based features (night transactions, time categories)
  - Account age features (new account flags, age categories)
  - Fraud history risk levels
  - Transaction ratios and velocities

### 🤖 Model Training & Evaluation
- **Multiple Models**: Support for Random Forest, Logistic Regression, Gradient Boosting, and SVM
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, and AUC
- **Visualizations**: 
  - Model performance comparison charts
  - ROC curves for binary classification
  - Confusion matrices for each model
- **Model Comparison**: Side-by-side comparison of all trained models

### 🔮 Interactive Predictions
- **Real-time Predictions**: Make predictions on new data using trained models
- **Dynamic Input Forms**: Automatically generated input forms based on features
- **Confidence Scores**: View prediction probabilities for each class
- **Visual Results**: Clear fraud/legitimate indicators with confidence levels

### 📈 Feature Analysis
- **Feature Importance**: View most important features for tree-based models
- **Correlation Analysis**: Analyze feature correlations with target variable
- **Interactive Charts**: Dynamic visualizations for feature exploration

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ensure Your Dataset
Make sure your fraud detection dataset (`final_file.csv`) is in the project directory with the following expected columns:
- `Transaction_Amount`
- `Time_of_Transaction` 
- `Account_Age`
- `Previous_Fraudulent_Transactions`
- `Number_of_Transactions_Last_24H`
- `Fraudulent` (target column)

### 3. Run the Dashboard
```bash
streamlit run learning.py
```

## 📋 How to Use

### Step 1: Data Upload & Analysis
1. Navigate to "Data Upload & Analysis"
2. Either upload a CSV file or click "Use Default Dataset"
3. Select the target column (usually "Fraudulent")
4. Review dataset statistics and class distribution

### Step 2: Data Preprocessing  
1. Go to "Data Preprocessing"
2. Configure preprocessing parameters:
   - **Scaling Method**: Standard (recommended) or MinMax
   - **Encoding Method**: OneHot (recommended) or Label
   - **Imbalance Handling**: SMOTE (recommended), undersample, SMOTETomek, or none
   - **Test Size**: Proportion for test split (default: 0.2)
   - **Generate Features**: Enable automatic feature engineering
3. Click "Run Preprocessing" and wait for completion

### Step 3: Model Training
1. Navigate to "Model Training & Evaluation"
2. Select models to train (Random Forest and Logistic Regression recommended)
3. Click "Train Models" and wait for completion
4. Review model performance metrics and visualizations

### Step 4: Make Predictions
1. Go to "Predictions"
2. Select a trained model
3. Input feature values using the generated form
4. Click "Make Prediction" to see results

### Step 5: Feature Analysis
1. Navigate to "Feature Analysis"
2. Select a model for analysis
3. Explore feature importance and correlations

## 🔧 Integration with Preprocessing Module

The dashboard fully integrates your `preprocessing.py` module:

```python
from preprocessing import DataPreprocessor, preprocess_data, get_data_summary

# Using the convenience function
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
    df=df,
    target='Fraudulent',
    scaling_method='standard',
    encoding_method='onehot',
    imbalance_method='smote',
    test_size=0.2,
    generate_new_features=True
)

# Using the class directly
preprocessor = DataPreprocessor(
    scaling_method='standard',
    encoding_method='onehot',
    imbalance_method='smote'
)
X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df, 'Fraudulent')
```

## 📊 Generated Features

The dashboard automatically creates additional features for improved fraud detection:

- **Amount Features**: Log transformations, percentiles
- **Time Features**: Night transaction flags, time categories  
- **Activity Features**: High transaction activity indicators
- **Account Features**: New account flags, age categories
- **Risk Features**: Fraud history levels, risk scores
- **Ratio Features**: Amount per transaction, spending velocity

## 📁 File Structure
```
project-streamlit/
├── learning.py              # Main dashboard application
├── preprocessing.py         # Preprocessing module
├── final_file.csv          # Default fraud dataset
├── requirements.txt        # Package dependencies
├── README.md              # This file
└── .venv/                 # Python virtual environment
```

## 🎯 Key Advantages

1. **No Code Preprocessing**: Visual interface for complex data preprocessing
2. **Multiple ML Models**: Compare different algorithms easily
3. **Real-time Predictions**: Test new transactions instantly  
4. **Feature Engineering**: Automatic creation of fraud-specific features
5. **Comprehensive Analysis**: End-to-end fraud detection pipeline
6. **Export Ready**: Download processed datasets for further analysis

## 🐛 Troubleshooting

- **Missing packages**: Run `pip install -r requirements.txt`
- **Dataset not found**: Ensure `final_file.csv` is in the project directory
- **Memory issues**: Try with smaller datasets or reduce feature generation
- **Model training slow**: Start with fewer models or smaller datasets

## 📝 Tips for Best Results

1. **Class Imbalance**: Use SMOTE for imbalanced fraud datasets
2. **Feature Scaling**: Use StandardScaler for most ML algorithms
3. **Encoding**: OneHot encoding works better with tree-based models
4. **Feature Engineering**: Enable for better fraud detection performance
5. **Model Selection**: Random Forest typically performs well for fraud detection

## 🤝 Contributing

Feel free to enhance the dashboard with additional:
- ML models (XGBoost, Neural Networks)
- Visualization options
- Feature engineering techniques  
- Export formats
- Advanced analysis features