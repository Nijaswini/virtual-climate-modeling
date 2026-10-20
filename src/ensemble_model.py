
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "outputs" / "cleaned_temperature.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

for month in months:
    df[month] = pd.to_numeric(df[month], errors="coerce")

df["Valid_Months"] = df[months].notna().sum(axis=1)
df = df[df["Valid_Months"] == 12].copy()
df["Annual_Mean_Anomaly"] = df[months].mean(axis=1)
df = df.sort_values("Year").reset_index(drop=True)

df["Lag_1"] = df["Annual_Mean_Anomaly"].shift(1)
df["Lag_2"] = df["Annual_Mean_Anomaly"].shift(2)
df["Rolling_3yr_Mean"] = (
    df["Annual_Mean_Anomaly"].shift(1).rolling(3).mean()
)
df = df.dropna().reset_index(drop=True)

features = ["Year", "Lag_1", "Lag_2", "Rolling_3yr_Mean"]
X = df[features]
y = df["Annual_Mean_Anomaly"]

splitter = TimeSeriesSplit(n_splits=5)
records = []

for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit models using only the training period
    linear = LinearRegression()
    forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )

    linear.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    linear_pred = linear.predict(X_test)
    forest_pred = forest.predict(X_test)

    # Equal-weight ensemble
    ensemble_pred = 0.5 * linear_pred + 0.5 * forest_pred

    # Previous-year baseline
    baseline_pred = X_test["Lag_1"].to_numpy()

    predictions = {
        "Previous-year baseline": baseline_pred,
        "Linear Regression": linear_pred,
        "Random Forest": forest_pred,
        "Equal-weight ensemble": ensemble_pred
    }

    for name, pred in predictions.items():
        records.append({
            "Fold": fold,
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5
        })

results = pd.DataFrame(records)

print("\nResults by fold:")
print(results.round(4).to_string(index=False))

summary = (
    results.groupby("Model")[["MAE", "RMSE"]]
    .mean()
    .sort_values("MAE")
    .reset_index()
)

print("\nAverage performance across folds:")
print(summary.round(4).to_string(index=False))

output_file = OUTPUT_DIR / "ensemble_comparison.csv"
results.to_csv(output_file, index=False)

summary_file = OUTPUT_DIR / "ensemble_summary.csv"
summary.to_csv(summary_file, index=False)

print("\nSaved fold results:", output_file)
print("Saved model summary:", summary_file)
