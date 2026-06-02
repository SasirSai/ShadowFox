import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, explained_variance_score, max_error
import os

def main():
    # Load the data
    file_path = os.path.join(os.path.dirname(__file__), 'HousingData.csv')
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    # SimpleImputer fills missing values with the median of each column
    # StandardScaler normalizes the features
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define the models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Evaluate models
    best_model = None
    best_r2 = -float('inf')
    best_name = ""

    print("\n--- Initial Model Evaluation ---")
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.4f}")
        print(f"  MedAE: {medae:.4f}")
        print(f"  Explained Variance: {evs:.4f}")
        print(f"  Max Error: {max_err:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_name = name

    print(f"\n==========================================")
    print(f"Most Precise Initial Model: {best_name}")
    print(f"Best R2 Score: {best_r2:.4f}")
    print(f"==========================================")

    # Fine-tune the best model
    if best_name in ['Random Forest', 'Gradient Boosting']:
        print(f"\n--- Fine-tuning {best_name} ---")
        if best_name == 'Random Forest':
            param_grid = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            }
        else:
            param_grid = {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__learning_rate': [0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 4, 5]
            }
            
        grid_search = GridSearchCV(
            Pipeline([('preprocessor', preprocessor), ('regressor', models[best_name])]),
            param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_tuned_model = grid_search.best_estimator_
        y_pred_tuned = best_tuned_model.predict(X_test)
        
        tuned_mse = mean_squared_error(y_test, y_pred_tuned)
        tuned_rmse = np.sqrt(tuned_mse)
        tuned_mae = mean_absolute_error(y_test, y_pred_tuned)
        tuned_mape = mean_absolute_percentage_error(y_test, y_pred_tuned)
        tuned_medae = median_absolute_error(y_test, y_pred_tuned)
        tuned_evs = explained_variance_score(y_test, y_pred_tuned)
        tuned_max_err = max_error(y_test, y_pred_tuned)
        tuned_r2 = r2_score(y_test, y_pred_tuned)
        
        print(f"\nTuned {best_name} Results:")
        print(f"  Best Parameters: {grid_search.best_params_}")
        print(f"  Tuned MSE: {tuned_mse:.4f}")
        print(f"  Tuned RMSE: {tuned_rmse:.4f}")
        print(f"  Tuned MAE:  {tuned_mae:.4f}")
        print(f"  Tuned MAPE: {tuned_mape:.4f}")
        print(f"  Tuned MedAE: {tuned_medae:.4f}")
        print(f"  Tuned Explained Variance: {tuned_evs:.4f}")
        print(f"  Tuned Max Error: {tuned_max_err:.4f}")
        print(f"  Tuned R2 Score: {tuned_r2:.4f}")
        
        # Feature Importances
        print(f"\n--- Feature Importances ({best_name}) ---")
        importances = best_tuned_model.named_steps['regressor'].feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print(importance_df.to_string(index=False))

        # Save predictions for external visualization
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_tuned})
        results_file = os.path.join(os.path.dirname(__file__), 'predictions.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\nPredictions saved to '{results_file}' for external visualization.")
        
        # Determine the absolute best model
        final_r2 = max(best_r2, tuned_r2)
        print(f"\nFinal Best R2 Score for Prediction: {final_r2:.4f}")

if __name__ == '__main__':
    main()
