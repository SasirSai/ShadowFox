# Boston House Price Prediction: Detailed Walkthrough

This walkthrough outlines the complete pipeline built for predicting the median values of owner-occupied homes (`MEDV`) in Boston. The project has been deliberately structured into two completely separate scripts to decouple heavy machine learning computation from visualization rendering.

## 🏗️ Architecture & Separation of Concerns

To keep the codebase modular, the project is split into two independent files:
1. `house_price_prediction.py`: The "brain" of the project. It handles data cleaning, trains the machine learning models, calculates deep analytics, and saves the final predictions.
2. `visualizations.py`: The "artist" of the project. It reads the finalized predictions from the first script and generates analytical graphs.

The bridge between these two files is a lightweight CSV file named `predictions.csv`.

---

## 1. Machine Learning Pipeline (`house_price_prediction.py`)

This script is responsible for building a model capable of high-precision regression.

### A. Data Preprocessing
- **Handling Missing Values**: We use `SimpleImputer(strategy='median')` to fill any `NA` gaps in the dataset without letting extreme outliers skew the averages.
- **Feature Scaling**: `StandardScaler` forces all features (from crime rates to tax rates) into a standard format with a mean of 0 and variance of 1. This prevents larger numbers from unfairly dominating the model's math.

### B. Model Selection & Tuning
The script tests three models: **Linear Regression**, **Random Forest**, and **Gradient Boosting**. 
Gradient Boosting proved to be the most accurate. To squeeze out every drop of precision, we pass it through `GridSearchCV`. This algorithm cross-validates multiple combinations of "learning rates" and "tree depths" to find the absolute mathematically optimal model.

### C. Exhaustive Evaluation Analytics
Rather than basic metrics, the script calculates an exhaustive list of performance indicators:
- **MAPE (~10.1%)**: The percentage by which our predictions are off.
- **MedAE (~1.44)**: Shows that 50% of all predictions are within $1,440 of the true price.
- **Explained Variance (~0.898)**: Confirms the model captures roughly 90% of the statistical variance in house prices.

### D. Extracting Feature Importances
Decision trees (like Gradient Boosting) inherently track which features were most useful when making splits. The script extracts the `feature_importances_` array to objectively prove what drives Boston housing prices. 
*Result*: The **Number of Rooms (RM)** and the **Socioeconomic Status (LSTAT)** account for over 75% of a home's value in this dataset.

---

## 2. The Handoff (`predictions.csv`)

> [!TIP]
> **Why export to CSV?**
> By exporting `y_test` (actual prices) and `y_pred` (predicted prices) to `predictions.csv`, we never have to retrain the machine learning model just to tweak a chart's color or font.

At the very end of the ML script, a Pandas DataFrame saves the exact test labels alongside our tuned model's predictions into `predictions.csv`. 

---

## 3. Visualization Generation (`visualizations.py`)

This standalone script strictly focuses on rendering the analytics graphically using `matplotlib` and `seaborn`. It loads `predictions.csv` and outputs two critical PNG charts.

### A. Scatterplot: Actual vs. Predicted Prices
- **What it is**: `actual_vs_predicted.png`
- **How it's made**: A Seaborn `scatterplot` plots the True prices on the X-axis against our Predicted prices on the Y-axis. 
- **The Red Line**: We dynamically draw a dashed red line (`plt.plot(..., 'r--')`) representing a perfect 1:1 prediction. The closer our blue scatter dots hug this red line, the more accurate the model is.

### B. Histogram: Distribution of Residuals
- **What it is**: `residuals_distribution.png`
- **How it's made**: The script calculates the **Residuals** (the literal mathematical difference between the actual price and the predicted price). It then uses a Seaborn `histplot` with a Kernel Density Estimate (`kde=True`) to plot the frequency of these errors.
- **Interpretation**: A perfectly unbiased model will result in a bell curve centered perfectly at `0.0` (meaning most errors are negligible, with no bias toward constantly over-predicting or under-predicting).

> [!NOTE]
> To regenerate the graphs with different visual styling, you only need to run `python visualizations.py`. You do not need to wait for the machine learning models to retrain!
