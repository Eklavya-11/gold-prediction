import pandas as pd
import numpy as np
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime, date, timedelta

print("Starting forecasting API...")

# =========================
# Constants & Model Loading
# =========================
MODEL_FILE = "model_files/xgboost_model.pkl"
FEATURES_FILE = "model_files/features_list.pkl"

if not os.path.exists(MODEL_FILE):
    print("Warning: XGBoost model not found. Please run train_xgboost.py first.")
    model = None
    features_list = None
else:
    model = joblib.load(MODEL_FILE)
    features_list = joblib.load(FEATURES_FILE)
    print("XGBoost model loaded successfully.")

# Constants
TROY_OUNCE_TO_10G = 10 / 31.10348
INDIA_IMPORT_DUTY = 1.15
K22_RATIO = 0.916

# =========================
# FastAPI App
# =========================
app = FastAPI(title="Gold Price Prediction API - Luxury Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    year: int
    month: int
    day: int

def get_macro_data_for_date(target_date: date = None):
    """Fetches macro data. If target_date is none, fetches latest. Otherwise fetches for that date (or closest prior)."""
    tickers = {
        'Global_Gold_USD': 'GC=F',
        'Crude_Oil_USD': 'CL=F',
        'SP500': '^GSPC',
        'USD_INR': 'INR=X'
    }
    
    if target_date is None:
        end_date = datetime.now()
    else:
        # Fetch up to the target date + 1 day to ensure we capture the target date
        end_date = datetime(target_date.year, target_date.month, target_date.day) + timedelta(days=1)
        
    start_date = end_date - timedelta(days=15) # Buffer for rolling means
    
    macro_dfs = []
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close'].copy()
        else:
            df = df[['Close']].copy()
        df.columns = [name]
        macro_dfs.append(df)
        
    merged = pd.concat(macro_dfs, axis=1).ffill().dropna()
    
    # Calculate lags and rolling on the recent data
    merged['gold_lag_1'] = merged['Global_Gold_USD'].shift(1)
    merged['gold_lag_3'] = merged['Global_Gold_USD'].shift(3)
    merged['gold_roll_7'] = merged['Global_Gold_USD'].rolling(window=7).mean()
    merged['oil_roll_7'] = merged['Crude_Oil_USD'].rolling(window=7).mean()
    merged['sp500_roll_7'] = merged['SP500'].rolling(window=7).mean()
    
    # Return the last row available (closest to target_date)
    return merged.iloc[-1].to_dict()

def get_actual_price(macro_data: dict):
    # Formula: (Global Gold USD / 31.10348) * 10 * USD_INR * 1.15
    gold_usd = macro_data.get('Global_Gold_USD', 0)
    usd_inr = macro_data.get('USD_INR', 0)
    if gold_usd and usd_inr:
        return gold_usd * TROY_OUNCE_TO_10G * usd_inr * INDIA_IMPORT_DUTY
    return None

@app.post("/predict")
def predict_price(data: PredictionRequest):
    if model is None:
        return {"error": "Model is not trained yet."}
        
    try:
        req_date = date(data.year, data.month, data.day)
        is_historical = req_date < date.today()
        
        # Use latest macro data for future, or historical macro data for past
        macro_data = get_macro_data_for_date(req_date if is_historical else None)
        
        # Build the feature dictionary for prediction
        feature_dict = {
            'year': data.year,
            'month': data.month,
            'day': data.day,
            **macro_data
        }
        
        # Create DataFrame in the exact order the model expects
        df_features = pd.DataFrame([feature_dict])[features_list]
        
        # Generate Prediction
        prediction_24k = float(model.predict(df_features)[0])
        prediction_22k = prediction_24k * K22_RATIO
        
        response = {
            "predicted_price_24k": round(prediction_24k, 2),
            "predicted_price_22k": round(prediction_22k, 2),
            "features_used": {
                "Global_Gold_USD": round(macro_data.get("Global_Gold_USD", 0), 2),
                "USD_INR": round(macro_data.get("USD_INR", 0), 2),
                "Crude_Oil_USD": round(macro_data.get("Crude_Oil_USD", 0), 2),
                "SP500": round(macro_data.get("SP500", 0), 2)
            }
        }
        
        if is_historical:
            actual_24k = get_actual_price(macro_data)
            if actual_24k:
                actual_22k = actual_24k * K22_RATIO
                response["actual_price_24k"] = round(actual_24k, 2)
                response["actual_price_22k"] = round(actual_22k, 2)
        
        return response
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/historical_macro")
def get_historical_macro():
    """Endpoint to provide historical macro data for the dashboard."""
    try:
        if os.path.exists('data/merged_Data.csv'):
            df = pd.read_csv('data/merged_Data.csv')
            df = df.tail(30)
            return df[['date', 'price', 'Global_Gold_USD', 'USD_INR', 'SP500', 'Crude_Oil_USD']].to_dict(orient='records')
        return {"error": "No historical data found."}
    except Exception as e:
        return {"error": str(e)}
