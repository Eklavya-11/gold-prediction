import pandas as pd
import yfinance as yf
import numpy as np
import os

def fetch_macro_data(start_date, end_date):
    """
    Fetches macroeconomic data from Yahoo Finance.
    - Global Gold Price (GC=F) in USD/oz
    - Crude Oil (CL=F)
    - S&P 500 (^GSPC)
    - USD to INR (INR=X)
    """
    tickers = {
        'Global_Gold_USD': 'GC=F',
        'Crude_Oil_USD': 'CL=F',
        'SP500': '^GSPC',
        'USD_INR': 'INR=X'
    }
    
    data_frames = []
    
    for name, ticker in tickers.items():
        print(f"Fetching {name} ({ticker})...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close'].copy()
        else:
            df = df[['Close']].copy()
            
        df.columns = [name]
        data_frames.append(df)
        
    merged_macro = pd.concat(data_frames, axis=1)
    
    # Forward fill missing weekend data
    merged_macro = merged_macro.ffill().dropna()
    merged_macro = merged_macro.reset_index()
    merged_macro['Date'] = pd.to_datetime(merged_macro['Date']).dt.date
    
    return merged_macro

def build_features(df):
    """Feature engineering: rolling averages and lags."""
    df = df.sort_values('Date').copy()
    
    df['gold_lag_1'] = df['Global_Gold_USD'].shift(1)
    df['gold_lag_3'] = df['Global_Gold_USD'].shift(3)
    
    df['gold_roll_7'] = df['Global_Gold_USD'].rolling(window=7).mean()
    df['oil_roll_7'] = df['Crude_Oil_USD'].rolling(window=7).mean()
    df['sp500_roll_7'] = df['SP500'].rolling(window=7).mean()
    
    return df.dropna()

def main():
    print("Fetching historical global data for the past 5 years...")
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=5*365)
    
    df_macro = fetch_macro_data(start_date, end_date)
    
    # Generate Indian Gold Price (INR per 10 grams)
    # Formula: (Global Gold USD / 31.10348) * 10 * USD_INR * 1.15 (15% import duty premium)
    TROY_OUNCE_TO_10G = 10 / 31.10348
    INDIA_IMPORT_DUTY_MULTIPLIER = 1.15
    
    df_macro['price'] = (
        df_macro['Global_Gold_USD'] 
        * TROY_OUNCE_TO_10G 
        * df_macro['USD_INR'] 
        * INDIA_IMPORT_DUTY_MULTIPLIER
    )
    
    # Add time features
    df_macro['date'] = pd.to_datetime(df_macro['Date'])
    df_macro['year'] = df_macro['date'].dt.year
    df_macro['month'] = df_macro['date'].dt.month
    df_macro['day'] = df_macro['date'].dt.day
    
    print("Building engineered features...")
    df_merged = build_features(df_macro)
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/merged_Data.csv'
    df_merged.to_csv(out_path, index=False)
    print(f"Successfully generated India Gold dataset and saved to {out_path} with shape {df_merged.shape}")

if __name__ == "__main__":
    main()
