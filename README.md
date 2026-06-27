# gold-prediction

Hey! This is a machine learning project I built to forecast gold prices. It uses historical macroeconomic data (like global spot prices, crude oil, S&P 500, and USD/ exchange rates) to predict both 24K and 22K gold prices.

I wanted to make something that improves my understaning, so if you select a date from the past, the app will fetch the actual historical price for that day and compare it with the model's prediction to show the exact error margin. 

### What's Inside

- **`prediction_model/`**: The Python backend. Uses FastAPI to serve an XGBoost model. It handles fetching real-time data from Yahoo Finance and engineering the features on the fly.
- **`webUI/`**: The frontend dashboard built with Next.js, React, and Tailwind CSS. It uses a clean Catppuccin (Latte) white theme.
- **`notebooks/`**: A Jupyter Notebook (`01_Initial_Working_Testing.ipynb`) showing my initial data exploration (EDA), feature engineering, and model evaluation (comparing XGBoost vs Linear Regression).

### How to Run Locally

You can spin everything up using Docker Compose:

```bash
docker-compose up --build
```

If you want to run it manually without Docker:

**1. Start Backend:**
```bash
cd prediction_model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

**2. Start Frontend:**
```bash
cd webUI
npm install
npm run dev
```
Then just open `http://localhost:3000` in your browser. 

## Screenshots

