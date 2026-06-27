import pandas as pd
import numpy as np

ORIGIN_DATE = pd.Timestamp("2019-01-01")

def build_time_features(X):
    X = X.copy()

    X["date"] = pd.to_datetime(
        dict(year=X.year, month=X.month, day=X.day)
    )

    X["t"] = (X["date"] - ORIGIN_DATE).dt.days

    X["sin_year"] = np.sin(2 * np.pi * X["t"] / 365.25)
    X["cos_year"] = np.cos(2 * np.pi * X["t"] / 365.25)

    X["week_of_month"] = ((X["day"] - 1) // 7) + 1
    X["sin_wom"] = np.sin(2 * np.pi * X["week_of_month"] / 4)
    X["cos_wom"] = np.cos(2 * np.pi * X["week_of_month"] / 4)

    return X[["t", "sin_year", "cos_year", "sin_wom", "cos_wom"]]
