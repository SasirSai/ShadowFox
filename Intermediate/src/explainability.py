import shap
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SHAPExplainer:
    def __init__(self, models_dir=None, data_dir=None):
        base_dir = os.path.dirname(__file__)
        if models_dir is None:
            models_dir = os.path.join(base_dir, '../models')
        if data_dir is None:
            data_dir = os.path.join(base_dir, '../data')
            
        self.model = joblib.load(os.path.join(models_dir, 'best_model.pkl'))
        self.feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        
        try:
            self.model_name = joblib.load(os.path.join(models_dir, 'best_model_name.pkl'))
        except:
            self.model_name = "Unknown"
            
        # Load background data for SHAP explainer
        try:
            self.X_train_bg = pd.read_csv(os.path.join(data_dir, 'X_train_preprocessed.csv'))
        except:
            self.X_train_bg = None
            
        # Initialize explainer
        if self.model_name in ['Random Forest', 'XGBoost', 'Decision Tree']:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            if self.X_train_bg is not None:
                bg_summary = shap.kmeans(self.X_train_bg, min(10, len(self.X_train_bg)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, bg_summary)
            else:
                self.explainer = None
                
    def get_shap_values(self, X_instance):
        if not self.explainer:
            return None
            
        if isinstance(self.explainer, shap.TreeExplainer):
            shap_values = self.explainer.shap_values(X_instance)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Positive class
            elif len(shap_values.shape) == 3: # Some versions return (N, M, C)
                shap_values = shap_values[:, :, 1]
            return shap_values
        else:
            shap_values = self.explainer.shap_values(X_instance)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            return shap_values
            
    def get_summary_plot(self):
        if self.X_train_bg is None or not self.explainer:
            return None
        
        shap_values = self.get_shap_values(self.X_train_bg)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_train_bg, feature_names=self.feature_names, show=False)
        fig = plt.gcf()
        return fig
        
    def get_waterfall_plot(self, X_instance):
        if not self.explainer:
            return None
            
        if isinstance(self.explainer, shap.TreeExplainer):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                
        raw_shap = self.get_shap_values(X_instance)[0]
        if len(raw_shap.shape) == 2:
            shap_values = raw_shap[:, 1] if raw_shap.shape[1] > 1 else raw_shap[:, 0]
        else:
            shap_values = raw_shap
            
        plt.figure(figsize=(8, 5))
        # Create a single Explanation object for the waterfall plot
        explanation = shap.Explanation(values=shap_values, 
                                       base_values=expected_value, 
                                       data=X_instance[0], 
                                       feature_names=self.feature_names)
                                       
        shap.waterfall_plot(explanation, show=False)
        fig = plt.gcf()
        return fig

    def get_human_readable_explanation(self, X_instance, prediction, confidence):
        raw_shap = self.get_shap_values(X_instance)[0]
        if len(raw_shap.shape) == 2:
            shap_values = raw_shap[:, 1] if raw_shap.shape[1] > 1 else raw_shap[:, 0]
        else:
            shap_values = raw_shap
        
        # Safe extraction for array-wrapped values
        def safe_val(val):
            if isinstance(val, (np.ndarray, list)):
                return np.ravel(val)[0]
            return val
            
        # Zip features with their SHAP values
        feature_importance = list(zip(self.feature_names, shap_values, X_instance[0]))
        # Sort by absolute SHAP value to find most impactful features
        feature_importance.sort(key=lambda x: abs(safe_val(x[1])), reverse=True)
        
        top_positive = [f for f in feature_importance if safe_val(f[1]) > 0][:3]
        top_negative = [f for f in feature_importance if safe_val(f[1]) < 0][:3]
        
        explanation = f"**Loan {prediction}** with {confidence}% confidence.\n\n"
        
        if prediction == "Approved":
            explanation += "**Key Reasons for Approval:**\n"
            for f, val, x_val in top_positive:
                explanation += f"- Strong contribution from {f} (Value: {round(x_val, 2) if isinstance(x_val, float) else x_val})\n"
            if top_negative:
                explanation += "\n**Risk Factors to Monitor:**\n"
                for f, val, x_val in top_negative:
                    explanation += f"- {f} negatively impacted the score.\n"
        else:
            explanation += "**Key Reasons for Rejection:**\n"
            for f, val, x_val in top_negative:
                explanation += f"- High risk indicated by {f} (Value: {round(x_val, 2) if isinstance(x_val, float) else x_val})\n"
            if top_positive:
                explanation += "\n**Positive Factors:**\n"
                for f, val, x_val in top_positive:
                    explanation += f"- {f} was favorable but insufficient.\n"
                    
        return explanation
