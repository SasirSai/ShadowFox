import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Create meaningful financial intelligence features.
    """
    df = df.copy()
    
    # 1. Income Strength Score (Total Income)
    df['Income_Strength_Score'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    # 2. Loan Burden Ratio (Loan Amount / Total Income)
    # Note: LoanAmount is usually in thousands in this dataset, but we'll use it as is for ratio.
    # Add 1 to denominator to avoid division by zero.
    df['Loan_Burden_Ratio'] = df['LoanAmount'] / (df['Income_Strength_Score'] + 1)
    
    # 3. Credit Reliability Score
    # Based on Credit history and Income stability
    # Assume salaried (Self_Employed == 'No') is more stable (score 1) than self-employed (score 0.8)
    if 'Self_Employed' in df.columns:
        income_stability = np.where(df['Self_Employed'] == 'No', 1.0, 0.8)
        df['Credit_Reliability_Score'] = df['Credit_History'] * income_stability
    else:
        df['Credit_Reliability_Score'] = df['Credit_History']
        
    # 4. Financial Stability Index
    # Combination of Education, Income, and Employment indicators
    education_score = np.where(df['Education'] == 'Graduate', 1.0, 0.5) if 'Education' in df.columns else 1.0
    employment_score = np.where(df['Self_Employed'] == 'No', 1.0, 0.8) if 'Self_Employed' in df.columns else 1.0
    income_score = np.log1p(df['Income_Strength_Score']) # Log transform to normalize
    
    df['Financial_Stability_Index'] = education_score * employment_score * income_score
    
    # Optional: Log transformations of skewed variables to improve model performance
    df['ApplicantIncome_Log'] = np.log1p(df['ApplicantIncome'])
    df['LoanAmount_Log'] = np.log1p(df['LoanAmount'])
    
    return df
