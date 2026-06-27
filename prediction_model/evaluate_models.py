import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import os

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}, model

def main():
    print("Loading data for evaluation...")
    if not os.path.exists('data/merged_Data.csv'):
        print("Data not found. Please run data_pipeline.py first.")
        return
        
    df = pd.read_csv('data/merged_Data.csv').sort_values('date')
    
    target = 'price'
    features = [
        'Global_Gold_USD', 'Crude_Oil_USD', 'SP500', 'USD_INR',
        'gold_lag_1', 'gold_lag_3', 'gold_roll_7', 'oil_roll_7', 'sp500_roll_7',
        'year', 'month', 'day'
    ]
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    split_idx = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    models_to_evaluate = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, objective='reg:squarederror')
    }
    
    results = []
    trained_models = {}
    
    for name, model in models_to_evaluate.items():
        metrics, trained_model = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results.append(metrics)
        trained_models[name] = trained_model
        
    results_df = pd.DataFrame(results)
    
    # Identify the best model (lowest RMSE)
    best_model_name = results_df.loc[results_df['RMSE'].idxmin()]['Model']
    print(f"\nBest Model: {best_model_name}")
    
    # Retrain best model on full dataset
    best_model = trained_models[best_model_name]
    best_model.fit(X, y)
    
    os.makedirs('model_files', exist_ok=True)
    joblib.dump(best_model, 'model_files/xgboost_model.pkl') # keeping the same name for API compatibility
    joblib.dump(features, 'model_files/features_list.pkl')
    
    # Generate Markdown Report
    report = f"""# 🪙 Gold Price Predictor - Model Evaluation Report

## Dataset Overview
- **Target Variable:** Indian Gold Price (INR per 10 grams)
- **Features Used:** Global Gold Spot Price (USD/oz), USD/INR Exchange Rate, S&P 500, Crude Oil Prices, 7-day rolling averages, and lag variables.
- **Training Samples:** {len(X_train)}
- **Testing Samples:** {len(X_test)}

## Model Performance Comparison
| Model | MAE (₹) | RMSE (₹) | R² Score |
|-------|---------|----------|----------|
"""
    for _, row in results_df.iterrows():
        report += f"| {row['Model']} | ₹{row['MAE']:.2f} | ₹{row['RMSE']:.2f} | {row['R2']:.4f} |\n"
        
    report += f"""
## Conclusion
The best performing model was **{best_model_name}**. This model has been automatically saved and deployed to the backend API.
"""
    with open('model_report.md', 'w') as f:
        f.write(report)
        
    print("\nReport saved to model_report.md. You can share this on LinkedIn or GitHub!")

if __name__ == "__main__":
    main()
