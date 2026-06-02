import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    file_path = os.path.join(os.path.dirname(__file__), 'predictions.csv')
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please run house_price_prediction.py first.")
        return
        
    print(f"Loading predictions from {file_path}...")
    df = pd.read_csv(file_path)
    y_test = df['Actual']
    y_pred_tuned = df['Predicted']

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_tuned, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices (Tuned Model)')
    plt.tight_layout()
    actual_pred_path = os.path.join(os.path.dirname(__file__), 'actual_vs_predicted.png')
    plt.savefig(actual_pred_path)
    plt.close()
    
    # Residual Plot
    residuals = y_test - y_pred_tuned
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    resid_path = os.path.join(os.path.dirname(__file__), 'residuals_distribution.png')
    plt.savefig(resid_path)
    plt.close()
    
    print("\nVisualizations successfully generated and saved:")
    print(f"- {actual_pred_path}")
    print(f"- {resid_path}")

if __name__ == '__main__':
    generate_plots()
