import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class RiskAssessmentEngine:
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(__file__)
            models_dir = os.path.join(base_dir, '../models')
            
        self.model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        self.preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
        self.feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        
    def preprocess_input(self, input_dict):
        # Convert to DataFrame
        df = pd.DataFrame([input_dict])
        
        # Feature Engineering Steps
        # 1. Income Strength Score
        df['Income_Strength_Score'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # 2. Loan Burden Ratio
        df['Loan_Burden_Ratio'] = df['LoanAmount'] / (df['Income_Strength_Score'] + 1)
        
        # 3. Credit Reliability Score
        income_stability = 1.0 if df['Self_Employed'].values[0] == 'No' else 0.8
        df['Credit_Reliability_Score'] = df['Credit_History'] * income_stability
        
        # 4. Financial Stability Index
        education_score = 1.0 if df['Education'].values[0] == 'Graduate' else 0.5
        employment_score = 1.0 if df['Self_Employed'].values[0] == 'No' else 0.8
        income_score = np.log1p(df['Income_Strength_Score'].values[0])
        df['Financial_Stability_Index'] = education_score * employment_score * income_score
        
        # Log Transforms
        df['ApplicantIncome_Log'] = np.log1p(df['ApplicantIncome'])
        df['LoanAmount_Log'] = np.log1p(df['LoanAmount'])
        
        # Handle Dependents if needed
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].replace('3+', '3')
            df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')
        
        # Ensure all columns exist that were present during training (some might be dropped but preprocessor handles missing strictly)
        # For our case, preprocessor expects specific columns:
        # We need to make sure df has exactly the required columns before preprocessor.transform
        
        X_processed = self.preprocessor.transform(df)
        return X_processed, df
        
    def predict(self, input_dict):
        X_processed, df_engineered = self.preprocess_input(input_dict)
        
        # Predict probability of class 1 (Approval)
        if hasattr(self.model, "predict_proba"):
            prob_approve = self.model.predict_proba(X_processed)[0][1]
        else:
            pred = self.model.predict(X_processed)[0]
            prob_approve = 1.0 if pred == 1 else 0.0
            
        prediction = "Approved" if prob_approve >= 0.5 else "Rejected"
        confidence = max(prob_approve, 1 - prob_approve) * 100
        
        # Risk Score (0-100)
        # Higher probability of approval means lower risk
        risk_score = (1 - prob_approve) * 100
        
        if risk_score <= 25:
            risk_category = "Low Risk"
        elif risk_score <= 50:
            risk_category = "Medium Risk"
        elif risk_score <= 75:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"
            
        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "risk_score": round(risk_score, 2),
            "risk_category": risk_category,
            "probability_approve": prob_approve
        }
        
        self.log_prediction(input_dict, result)
        
        return result, X_processed
        
    def log_prediction(self, input_dict, result):
        base_dir = os.path.dirname(__file__)
        log_dir = os.path.join(base_dir, '../reports')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'prediction_logs.csv')
        
        log_data = input_dict.copy()
        log_data.update(result)
        log_data['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        df_log = pd.DataFrame([log_data])
        if not os.path.exists(log_file):
            df_log.to_csv(log_file, index=False)
        else:
            df_log.to_csv(log_file, mode='a', header=False, index=False)
