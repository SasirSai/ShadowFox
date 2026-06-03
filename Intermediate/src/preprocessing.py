import pandas as pd
import numpy as np

def load_data(filepath):
    """Load the dataset from the specified filepath."""
    return pd.read_csv(filepath)

def handle_missing_values(df):
    """
    Handle missing values:
    - Mode imputation for categorical
    - Median imputation for numerical
    """
    df = df.copy()
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Numerical columns
    num_cols = df.select_dtypes(exclude=['object']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    return df

def handle_outliers_iqr(df):
    """
    Handle outliers using the IQR technique.
    Caps outliers instead of dropping them to preserve dataset size.
    """
    df = df.copy()
    num_cols = df.select_dtypes(exclude=['object']).columns
    for col in num_cols:
        # Skip binary/categorical numerical columns
        if len(df[col].unique()) <= 10:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        
    return df

def clean_data(df):
    """
    Apply robust preprocessing pipeline:
    - Remove duplicates
    - Handle inconsistent categories (e.g. Dependents '3+' to '3')
    """
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle Dependents '3+'
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', '3')
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')
        
    # Drop Loan_ID as it's not predictive
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
        
    df = handle_outliers_iqr(df)
        
    return df
