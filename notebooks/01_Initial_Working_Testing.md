# Gold Prices Predictor - Initial Working & Testing

This notebook demonstrates the initial data exploration, feature engineering, and model testing for predicting Gold Prices in India.

We fetch global macroeconomic indicators and convert global spot gold prices into local INR prices per 10 grams, factoring in the standard 15% import duty.

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.linear_model import LinearRegression

# Set plotting style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)
```

## 1. Data Fetching and Conversion

We fetch data from the past 5 years using Yahoo Finance.

```python
# Define Tickers
tickers = {
    'Global_Gold_USD': 'GC=F',
    'Crude_Oil_USD': 'CL=F',
    'SP500': '^GSPC',
    'USD_INR': 'INR=X'
}

end_date = pd.Timestamp.now()
start_date = end_date - pd.Timedelta(days=5*365)

data_frames = []
for name, ticker in tickers.items():
    print(f"Fetching {name}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close'].copy()
    else:
        df = df[['Close']].copy()
    df.columns = [name]
    data_frames.append(df)
    
df_macro = pd.concat(data_frames, axis=1).ffill().dropna().reset_index()
df_macro['Date'] = pd.to_datetime(df_macro['Date'])
```

### Deriving the Indian Gold Price (₹)

Standard gold pricing is usually per 10 grams.
- 1 Troy Ounce = 31.10348 grams
- Import Duty / Premium ≈ 15%

```python
TROY_OUNCE_TO_10G = 10 / 31.10348
INDIA_IMPORT_DUTY = 1.15

df_macro['Gold_Price_INR_10g'] = (
    df_macro['Global_Gold_USD'] 
    * TROY_OUNCE_TO_10G 
    * df_macro['USD_INR'] 
    * INDIA_IMPORT_DUTY
)

df_macro.tail()
```

## 2. Exploratory Data Analysis (EDA)

Let's visualize the historical price of gold.

```python
plt.figure(figsize=(14, 7))
plt.plot(df_macro['Date'], df_macro['Gold_Price_INR_10g'], color='gold', linewidth=2)
plt.title('Historical Gold Price in India (INR per 10g)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (₹)', fontsize=12)
plt.show()
```

Let's look at the correlation matrix between our macroeconomic variables.

```python
corr = df_macro[['Gold_Price_INR_10g', 'Global_Gold_USD', 'Crude_Oil_USD', 'SP500', 'USD_INR']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

## 3. Feature Engineering

We'll add time-series features like rolling averages and lag features to help the model learn trends.

```python
df_features = df_macro.copy()

# Lags
df_features['gold_lag_1'] = df_features['Global_Gold_USD'].shift(1)
df_features['gold_lag_3'] = df_features['Global_Gold_USD'].shift(3)

# Rolling Means
df_features['gold_roll_7'] = df_features['Global_Gold_USD'].rolling(window=7).mean()
df_features['oil_roll_7'] = df_features['Crude_Oil_USD'].rolling(window=7).mean()
df_features['sp500_roll_7'] = df_features['SP500'].rolling(window=7).mean()

# Time features
df_features['year'] = df_features['Date'].dt.year
df_features['month'] = df_features['Date'].dt.month
df_features['day'] = df_features['Date'].dt.day

df_features = df_features.dropna()
```

## 4. Model Training & Evaluation

We split the data chronologically (80% training, 20% testing) to prevent data leakage from the future.

```python
features = [
    'Global_Gold_USD', 'Crude_Oil_USD', 'SP500', 'USD_INR',
    'gold_lag_1', 'gold_lag_3', 'gold_roll_7', 'oil_roll_7', 'sp500_roll_7',
    'year', 'month', 'day'
]
target = 'Gold_Price_INR_10g'

X = df_features[features]
y = df_features[target]

split_idx = int(len(df_features) * 0.8)
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")
```

### Baseline: Linear Regression

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print(f"Linear Regression MAE: ₹{mean_absolute_error(y_test, lr_preds):.2f}")
print(f"Linear Regression RMSE: ₹{np.sqrt(mean_squared_error(y_test, lr_preds)):.2f}")
print(f"Linear Regression R2: {r2_score(y_test, lr_preds):.4f}")
```

### Advanced: XGBoost

```python
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print(f"XGBoost MAE: ₹{mean_absolute_error(y_test, xgb_preds):.2f}")
print(f"XGBoost RMSE: ₹{np.sqrt(mean_squared_error(y_test, xgb_preds)):.2f}")
print(f"XGBoost R2: {r2_score(y_test, xgb_preds):.4f}")
```

## 5. Visualizing Predictions

```python
plt.figure(figsize=(14, 7))
plt.plot(df_features['Date'].iloc[split_idx:], y_test, label='Actual Price', color='gold', linewidth=2)
plt.plot(df_features['Date'].iloc[split_idx:], xgb_preds, label='XGBoost Prediction', color='green', linestyle='dashed')
plt.plot(df_features['Date'].iloc[split_idx:], lr_preds, label='Linear Reg Prediction', color='blue', linestyle='dotted')

plt.title('Actual vs Predicted Gold Prices', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (₹)', fontsize=12)
plt.legend()
plt.show()
```
