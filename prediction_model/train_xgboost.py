import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def main():
    print("Loading merged dataset...")
    df = pd.read_csv('data/merged_Data.csv')
    
    # Sort by date just to be sure
    df = df.sort_values('date')
    
    # Define features and target
    target = 'price'
    features = [
        'Global_Gold_USD', 'Crude_Oil_USD', 'SP500', 'USD_INR',
        'gold_lag_1', 'gold_lag_3', 'gold_roll_7', 'oil_roll_7', 'sp500_roll_7',
        'year', 'month', 'day'
    ]
    
    # Drop rows with NaN due to lagging/rolling
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    # Time Series Split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Initialize and train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f} NPR")
    print(f"RMSE: {rmse:.2f} NPR")
    
    # Retrain on full dataset for production
    print("Retraining on full dataset for production...")
    model.fit(X, y)
    
    # Save the model
    import os
    os.makedirs('model_files', exist_ok=True)
    joblib.dump(model, 'model_files/xgboost_model.pkl')
    
    # Also save the features list so the API knows what to expect
    joblib.dump(features, 'model_files/features_list.pkl')
    
    print("Model and features list saved to model_files/")

if __name__ == "__main__":
    main()
