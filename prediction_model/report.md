# Model Evaluation Report

## Dataset Overview
- **Target Variable:** Gold Price (INR per 10 grams)
- **Features Used:** Global Gold Spot Price (USD/oz), USD/INR Exchange Rate, S&P 500, Crude Oil Prices, 7-day rolling averages, and lag variables.
- **Training Samples:** 1034
- **Testing Samples:** 259

## Model Performance Comparison
| Model | MAE (₹) | RMSE (₹) | R² Score |
|-------|---------|----------|----------|
| Linear Regression | ₹3718.51 | ₹4591.04 | 0.9580 |
| Random Forest | ₹34391.32 | ₹40724.37 | -2.3016 |
| XGBoost | ₹34911.86 | ₹41237.59 | -2.3853 |

## Conclusion
The best performing model was **Linear Regression**. This model has been automatically saved and deployed to the backend API.
