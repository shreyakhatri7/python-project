"""
Preprocessing & Feature Engineering Module for Fraud Detection System
Member 2 — Preprocessing + Feature Engineering Specialist

This module provides automated data preprocessing capabilities:
- Automatic detection of numeric/categorical columns
- Missing value handling
- Categorical encoding (Label/OneHot)
- Numeric scaling (StandardScaler/MinMaxScaler)
- Train/test splitting
- Class imbalance handling (SMOTE/undersampling)
- Feature engineering (rolling averages, transaction ratios)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for fraud detection datasets.
    Automatically handles column detection, missing values, encoding, scaling,
    and class imbalance.
    """
    
    def __init__(self, scaling_method='standard', encoding_method='onehot', 
                 imbalance_method='smote', test_size=0.2, random_state=42):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        scaling_method : str, default='standard'
            Method for scaling numeric features: 'standard' or 'minmax'
        encoding_method : str, default='onehot'
            Method for encoding categorical features: 'label' or 'onehot'
        imbalance_method : str, default='smote'
            Method for handling class imbalance: 'smote', 'undersample', 'smotetomek', or None
        test_size : float, default=0.2
            Proportion of data to use for testing
        random_state : int, default=42
            Random state for reproducibility
        """
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.imbalance_method = imbalance_method
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize transformers (will be fitted during preprocessing)
        self.scaler = None
        self.encoders = {}
        self.onehot_encoder = None
        self.numeric_imputer = None
        self.categorical_imputer = None
        
        # Column tracking
        self.numeric_columns = []
        self.categorical_columns = []
        self.id_columns = []
        self.target_column = None
        self.feature_names = []
        
    def detect_column_types(self, df, target):
        """
        Automatically detect numeric, categorical, and ID columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target : str
            Name of the target column
            
        Returns:
        --------
        dict : Dictionary containing lists of column types
        """
        self.target_column = target
        
        # Exclude target from features
        feature_cols = [col for col in df.columns if col != target]
        
        self.numeric_columns = []
        self.categorical_columns = []
        self.id_columns = []
        
        for col in feature_cols:
            # Check if column is likely an ID column
            if 'id' in col.lower() or col.lower().endswith('_id'):
                self.id_columns.append(col)
                continue
            
            # Check data type
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Additional check: if unique values ratio is very high, might be ID
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.9 and df[col].nunique() > 100:
                    self.id_columns.append(col)
                else:
                    self.numeric_columns.append(col)
            else:
                self.categorical_columns.append(col)
        
        print("\n" + "="*60)
        print("COLUMN TYPE DETECTION")
        print("="*60)
        print(f"Target Column: {target}")
        print(f"Numeric Columns ({len(self.numeric_columns)}): {self.numeric_columns}")
        print(f"Categorical Columns ({len(self.categorical_columns)}): {self.categorical_columns}")
        print(f"ID Columns (excluded) ({len(self.id_columns)}): {self.id_columns}")
        
        return {
            'numeric': self.numeric_columns,
            'categorical': self.categorical_columns,
            'id': self.id_columns,
            'target': target
        }
    
    def analyze_missing_values(self, df):
        """
        Analyze missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame : Summary of missing values
        """
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        print("\n" + "="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        if len(missing_data) > 0:
            print(missing_data.to_string(index=False))
        else:
            print("No missing values found!")
        
        return missing_data
    
    def handle_missing_values(self, df):
        """
        Handle missing values in numeric and categorical columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame : DataFrame with imputed missing values
        """
        df_processed = df.copy()
        
        # Get ALL numeric columns (including newly generated ones)
        all_numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target if present
        if self.target_column and self.target_column in all_numeric_cols:
            all_numeric_cols.remove(self.target_column)
        # Exclude ID columns
        all_numeric_cols = [col for col in all_numeric_cols if col not in self.id_columns]
        
        # Get ALL categorical columns (including newly generated ones)
        all_categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        # Exclude ID columns
        all_categorical_cols = [col for col in all_categorical_cols if col not in self.id_columns]
        
        # Handle numeric missing values with median
        if all_numeric_cols:
            self.numeric_imputer = SimpleImputer(strategy='median')
            df_processed[all_numeric_cols] = self.numeric_imputer.fit_transform(
                df_processed[all_numeric_cols]
            )
        
        # Handle categorical missing values with most frequent
        if all_categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[all_categorical_cols] = self.categorical_imputer.fit_transform(
                df_processed[all_categorical_cols]
            )
        
        # Update column tracking to include newly generated columns
        self.numeric_columns = all_numeric_cols
        self.categorical_columns = all_categorical_cols
        
        print("\n" + "="*60)
        print("MISSING VALUE HANDLING")
        print("="*60)
        print(f"Numeric columns imputed with MEDIAN: {len(all_numeric_cols)}")
        print(f"Categorical columns imputed with MOST FREQUENT: {len(all_categorical_cols)}")
        print(f"Remaining missing values: {df_processed.isnull().sum().sum()}")
        
        return df_processed
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features using Label or OneHot encoding.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame : DataFrame with encoded categorical features
        """
        df_processed = df.copy()
        
        print("\n" + "="*60)
        print("CATEGORICAL ENCODING")
        print("="*60)
        print(f"Method: {self.encoding_method.upper()}")
        
        if not self.categorical_columns:
            print("No categorical columns to encode.")
            return df_processed
        
        if self.encoding_method == 'label':
            # Label Encoding
            for col in self.categorical_columns:
                self.encoders[col] = LabelEncoder()
                df_processed[col] = self.encoders[col].fit_transform(
                    df_processed[col].astype(str)
                )
                print(f"  {col}: {len(self.encoders[col].classes_)} unique values encoded")
        
        elif self.encoding_method == 'onehot':
            # One-Hot Encoding
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = self.onehot_encoder.fit_transform(df_processed[self.categorical_columns])
            encoded_columns = self.onehot_encoder.get_feature_names_out(self.categorical_columns)
            
            # Create encoded dataframe
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df_processed.index)
            
            # Drop original categorical columns and concat encoded ones
            df_processed = df_processed.drop(columns=self.categorical_columns)
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
            
            print(f"  Created {len(encoded_columns)} one-hot encoded features")
            for col in self.categorical_columns:
                col_features = [c for c in encoded_columns if c.startswith(col)]
                print(f"    {col}: {len(col_features)} categories")
        
        return df_processed
    
    def scale_numeric_features(self, X_train, X_test):
        """
        Scale numeric features using StandardScaler or MinMaxScaler.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
            
        Returns:
        --------
        tuple : (X_train_scaled, X_test_scaled)
        """
        print("\n" + "="*60)
        print("NUMERIC SCALING")
        print("="*60)
        print(f"Method: {self.scaling_method.upper()}")
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        # Get numeric columns that exist in the processed data
        numeric_cols_present = [col for col in self.numeric_columns if col in X_train.columns]
        
        if numeric_cols_present:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numeric_cols_present] = self.scaler.fit_transform(
                X_train[numeric_cols_present]
            )
            X_test_scaled[numeric_cols_present] = self.scaler.transform(
                X_test[numeric_cols_present]
            )
            
            print(f"  Scaled {len(numeric_cols_present)} numeric columns")
            return X_train_scaled, X_test_scaled
        else:
            print("  No numeric columns to scale.")
            return X_train, X_test
    
    def create_train_test_split(self, X, y):
        """
        Create train/test split.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print("\n" + "="*60)
        print("TRAIN/TEST SPLIT")
        print("="*60)
        print(f"Test Size: {self.test_size * 100}%")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Training target distribution:\n{y_train.value_counts().to_dict()}")
        print(f"Testing target distribution:\n{y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X_train, y_train):
        """
        Handle class imbalance using SMOTE, undersampling, or SMOTETomek.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
            
        Returns:
        --------
        tuple : (X_train_resampled, y_train_resampled)
        """
        print("\n" + "="*60)
        print("CLASS IMBALANCE HANDLING")
        print("="*60)
        
        # Check class distribution
        class_counts = y_train.value_counts()
        imbalance_ratio = class_counts.min() / class_counts.max()
        
        print(f"Original class distribution: {class_counts.to_dict()}")
        print(f"Imbalance ratio: {imbalance_ratio:.4f}")
        
        if self.imbalance_method is None:
            print("Method: None (keeping original distribution)")
            return X_train, y_train
        
        print(f"Method: {self.imbalance_method.upper()}")
        
        # Convert to numpy for imblearn
        X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
        
        if self.imbalance_method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif self.imbalance_method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state)
        elif self.imbalance_method == 'smotetomek':
            sampler = SMOTETomek(random_state=self.random_state)
        else:
            print(f"Unknown method: {self.imbalance_method}. Keeping original distribution.")
            return X_train, y_train
        
        X_resampled, y_resampled = sampler.fit_resample(X_train_values, y_train_values)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled, name=y_train.name)
        
        print(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
        print(f"Samples before: {len(y_train)}")
        print(f"Samples after: {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def generate_features(self, df, user_id_col=None):
        """
        Generate new features for fraud detection.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        user_id_col : str, optional
            Name of user ID column for grouping
            
        Returns:
        --------
        pd.DataFrame : DataFrame with additional features
        """
        df_features = df.copy()
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        new_features = []
        
        # Transaction Amount Features
        if 'Transaction_Amount' in df_features.columns:
            # Log transform of transaction amount
            df_features['Amount_Log'] = np.log1p(df_features['Transaction_Amount'])
            new_features.append('Amount_Log')
            
            # Amount percentile (relative to overall)
            df_features['Amount_Percentile'] = df_features['Transaction_Amount'].rank(pct=True)
            new_features.append('Amount_Percentile')
        
        # Time-based features
        if 'Time_of_Transaction' in df_features.columns:
            # Is nighttime transaction (between 10pm and 6am)
            df_features['Is_Night_Transaction'] = df_features['Time_of_Transaction'].apply(
                lambda x: 1 if (x >= 22 or x <= 6) else 0 if pd.notna(x) else 0
            )
            new_features.append('Is_Night_Transaction')
            
            # Time categories
            def categorize_time(hour):
                if pd.isna(hour):
                    return 'unknown'
                elif hour < 6:
                    return 'late_night'
                elif hour < 12:
                    return 'morning'
                elif hour < 18:
                    return 'afternoon'
                else:
                    return 'evening'
            
            df_features['Time_Category'] = df_features['Time_of_Transaction'].apply(categorize_time)
            if 'Time_Category' not in self.categorical_columns:
                self.categorical_columns.append('Time_Category')
            new_features.append('Time_Category')
        
        # Transaction frequency features
        if 'Number_of_Transactions_Last_24H' in df_features.columns:
            # High activity flag
            df_features['High_Transaction_Activity'] = (
                df_features['Number_of_Transactions_Last_24H'] > 
                df_features['Number_of_Transactions_Last_24H'].median()
            ).astype(int)
            new_features.append('High_Transaction_Activity')
        
        # Account Age features
        if 'Account_Age' in df_features.columns:
            # New account flag (less than 30 days)
            df_features['Is_New_Account'] = (df_features['Account_Age'] < 30).astype(int)
            new_features.append('Is_New_Account')
            
            # Account age categories
            df_features['Account_Age_Category'] = pd.cut(
                df_features['Account_Age'].fillna(-1),  # Temporary fill for binning
                bins=[-2, 0, 30, 90, 180, float('inf')],
                labels=['unknown', 'new', 'young', 'established', 'mature']
            ).astype(str)
            df_features.loc[df_features['Account_Age'].isna(), 'Account_Age_Category'] = 'unknown'
            if 'Account_Age_Category' not in self.categorical_columns:
                self.categorical_columns.append('Account_Age_Category')
            new_features.append('Account_Age_Category')
        
        # Fraud history risk
        if 'Previous_Fraudulent_Transactions' in df_features.columns:
            df_features['Has_Fraud_History'] = (
                df_features['Previous_Fraudulent_Transactions'] > 0
            ).astype(int)
            new_features.append('Has_Fraud_History')
            
            # Fraud risk level
            df_features['Fraud_Risk_Level'] = pd.cut(
                df_features['Previous_Fraudulent_Transactions'].fillna(-1),
                bins=[-2, -0.5, 0.5, 1.5, 2.5, float('inf')],
                labels=['unknown', 'none', 'low', 'medium', 'high']
            ).astype(str)
            df_features.loc[df_features['Previous_Fraudulent_Transactions'].isna(), 'Fraud_Risk_Level'] = 'unknown'
            if 'Fraud_Risk_Level' not in self.categorical_columns:
                self.categorical_columns.append('Fraud_Risk_Level')
            new_features.append('Fraud_Risk_Level')
        
        # Transaction amount to activity ratio
        if 'Transaction_Amount' in df_features.columns and 'Number_of_Transactions_Last_24H' in df_features.columns:
            df_features['Amount_Per_Transaction'] = (
                df_features['Transaction_Amount'] / 
                (df_features['Number_of_Transactions_Last_24H'] + 1)
            )
            new_features.append('Amount_Per_Transaction')
        
        # Amount relative to account age (spending velocity)
        if 'Transaction_Amount' in df_features.columns and 'Account_Age' in df_features.columns:
            df_features['Amount_To_Age_Ratio'] = (
                df_features['Transaction_Amount'] / 
                (df_features['Account_Age'] + 1)
            )
            new_features.append('Amount_To_Age_Ratio')
        
        print(f"Generated {len(new_features)} new features:")
        for feat in new_features:
            print(f"  - {feat}")
        
        return df_features
    
    def preprocess_data(self, df, target, generate_new_features=True):
        """
        Main preprocessing pipeline that performs all preprocessing steps.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target : str
            Name of the target column
        generate_new_features : bool, default=True
            Whether to generate new features
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, feature_names)
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING PIPELINE")
        print("="*60)
        print(f"Dataset shape: {df.shape}")
        print(f"Target column: {target}")
        
        # Step 1: Detect column types
        self.detect_column_types(df, target)
        
        # Step 2: Analyze missing values
        self.analyze_missing_values(df)
        
        # Step 3: Generate new features (before handling missing values)
        if generate_new_features:
            df_processed = self.generate_features(df)
        else:
            df_processed = df.copy()
        
        # Step 4: Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Step 5: Separate features and target
        # Drop ID columns and target
        cols_to_drop = self.id_columns + [target]
        X = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns])
        y = df_processed[target]
        
        # Step 6: Encode categorical features
        X = self.encode_categorical_features(X)
        
        # Step 7: Create train/test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(X, y)
        
        # Step 8: Scale numeric features
        X_train_scaled, X_test_scaled = self.scale_numeric_features(X_train, X_test)
        
        # Step 9: Handle class imbalance (only on training data)
        X_train_final, y_train_final = self.handle_class_imbalance(X_train_scaled, y_train)
        
        # Store feature names
        self.feature_names = list(X_train_final.columns)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Final training features shape: {X_train_final.shape}")
        print(f"Final testing features shape: {X_test_scaled.shape}")
        print(f"Total features: {len(self.feature_names)}")
        
        return X_train_final, X_test_scaled, y_train_final, y_test
    
    def get_feature_importance_ready_data(self):
        """
        Get feature names for feature importance analysis.
        
        Returns:
        --------
        list : List of feature names
        """
        return self.feature_names
    
    def transform_new_data(self, df):
        """
        Transform new data using fitted preprocessors.
        
        Parameters:
        -----------
        df : pd.DataFrame
            New dataframe to transform
            
        Returns:
        --------
        pd.DataFrame : Transformed dataframe
        """
        df_processed = df.copy()
        
        # Generate features if they were generated during training
        df_processed = self.generate_features(df_processed)
        
        # Handle missing values
        if self.numeric_imputer and self.numeric_columns:
            numeric_cols_present = [col for col in self.numeric_columns if col in df_processed.columns]
            df_processed[numeric_cols_present] = self.numeric_imputer.transform(
                df_processed[numeric_cols_present]
            )
        
        if self.categorical_imputer and self.categorical_columns:
            cat_cols_present = [col for col in self.categorical_columns if col in df_processed.columns]
            df_processed[cat_cols_present] = self.categorical_imputer.transform(
                df_processed[cat_cols_present]
            )
        
        # Drop ID columns
        df_processed = df_processed.drop(
            columns=[col for col in self.id_columns if col in df_processed.columns],
            errors='ignore'
        )
        
        # Encode categorical features
        if self.encoding_method == 'label':
            for col, encoder in self.encoders.items():
                if col in df_processed.columns:
                    df_processed[col] = encoder.transform(df_processed[col].astype(str))
        elif self.encoding_method == 'onehot' and self.onehot_encoder:
            cat_cols_present = [col for col in self.categorical_columns if col in df_processed.columns]
            if cat_cols_present:
                encoded_data = self.onehot_encoder.transform(df_processed[cat_cols_present])
                encoded_columns = self.onehot_encoder.get_feature_names_out(cat_cols_present)
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df_processed.index)
                df_processed = df_processed.drop(columns=cat_cols_present)
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
        
        # Scale numeric features
        if self.scaler:
            numeric_cols_present = [col for col in self.numeric_columns if col in df_processed.columns]
            if numeric_cols_present:
                df_processed[numeric_cols_present] = self.scaler.transform(
                    df_processed[numeric_cols_present]
                )
        
        return df_processed


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def preprocess_data(df, target, scaling_method='standard', encoding_method='onehot',
                    imbalance_method='smote', test_size=0.2, random_state=42,
                    generate_new_features=True):
    """
    Convenience function to preprocess data with a single function call.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Name of the target column
    scaling_method : str, default='standard'
        Method for scaling numeric features: 'standard' or 'minmax'
    encoding_method : str, default='onehot'
        Method for encoding categorical features: 'label' or 'onehot'
    imbalance_method : str, default='smote'
        Method for handling class imbalance: 'smote', 'undersample', 'smotetomek', or None
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random state for reproducibility
    generate_new_features : bool, default=True
        Whether to generate new features
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, preprocessor)
        Returns processed data splits and the fitted preprocessor object
    """
    preprocessor = DataPreprocessor(
        scaling_method=scaling_method,
        encoding_method=encoding_method,
        imbalance_method=imbalance_method,
        test_size=test_size,
        random_state=random_state
    )
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        df, target, generate_new_features=generate_new_features
    )
    
    return X_train, X_test, y_train, y_test, preprocessor


def quick_preprocess(df, target):
    """
    Quick preprocessing with default settings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Name of the target column
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, target)
    return X_train, X_test, y_train, y_test


def get_data_summary(df, target):
    """
    Get a comprehensive summary of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Name of the target column
        
    Returns:
    --------
    dict : Dictionary containing data summary
    """
    summary = {
        'shape': df.shape,
        'total_samples': len(df),
        'total_features': len(df.columns) - 1,
        'target_distribution': df[target].value_counts().to_dict(),
        'imbalance_ratio': df[target].value_counts().min() / df[target].value_counts().max(),
        'missing_values': df.isnull().sum().sum(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Shape: {summary['shape']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total features: {summary['total_features']}")
    print(f"Target distribution: {summary['target_distribution']}")
    print(f"Imbalance ratio: {summary['imbalance_ratio']:.4f}")
    print(f"Total missing values: {summary['missing_values']}")
    print(f"Numeric columns: {summary['numeric_columns']}")
    print(f"Categorical columns: {summary['categorical_columns']}")
    
    return summary


# ============================================================================
# MAIN EXECUTION (DEMO)
# ============================================================================

if __name__ == "__main__":
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('final_file.csv')
    
    # Display data summary
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    # Get data summary
    summary = get_data_summary(df, 'Fraudulent')
    
    # Option 1: Using the convenience function
    print("\n" + "="*60)
    print("OPTION 1: Using preprocess_data() function")
    print("="*60)
    
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df=df,
        target='Fraudulent',
        scaling_method='standard',  # or 'minmax'
        encoding_method='onehot',   # or 'label'
        imbalance_method='smote',   # or 'undersample', 'smotetomek', None
        test_size=0.2,
        generate_new_features=True
    )
    
    print("\n" + "="*60)
    print("FINAL OUTPUT")
    print("="*60)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"\nFeature names: {preprocessor.feature_names[:10]}...")  # First 10 features
    print(f"Total features: {len(preprocessor.feature_names)}")
    
    # Save processed data (optional)
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)
    
    # Combine X and y for saving
    train_data = X_train.copy()
    train_data['Fraudulent'] = y_train.values
    test_data = X_test.copy()
    test_data['Fraudulent'] = y_test.values
    
    train_data.to_csv('processed_train_data.csv', index=False)
    test_data.to_csv('processed_test_data.csv', index=False)
    
    print("Processed training data saved to: processed_train_data.csv")
    print("Processed testing data saved to: processed_test_data.csv")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
