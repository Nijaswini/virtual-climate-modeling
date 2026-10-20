
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "outputs" / "cleaned_temperature.csv"

df = pd.read_csv(DATA_FILE)

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

for month in months:
    df[month] = pd.to_numeric(df[month], errors="coerce")

# Use complete years only
df["Valid_Months"] = df[months].notna().sum(axis=1)
df = df[df["Valid_Months"] == 12].copy()
df["Annual_Mean_Anomaly"] = df[months].mean(axis=1)
df = df.sort_values("Year").reset_index(drop=True)

# Build features using past observations only
df["Lag_1"] = df["Annual_Mean_Anomaly"].shift(1)
df["Lag_2"] = df["Annual_Mean_Anomaly"].shift(2)
df["Rolling_3yr_Mean"] = (
    df["Annual_Mean_Anomaly"].shift(1).rolling(3).mean()
)
df = df.dropna().reset_index(drop=True)

features = ["Year", "Lag_1", "Lag_2", "Rolling_3yr_Mean"]
X = df[features]
y = df["Annual_Mean_Anomaly"]

models = {
    "Previous-year baseline": None,
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )
}

splitter = TimeSeriesSplit(n_splits=5)
records = []

for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, model in models.items():
        if name == "Previous-year baseline":
            pred = X_test["Lag_1"].to_numpy()
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        records.append({
            "Fold": fold,
            "Model": name,
            "Test_Start_Year": int(df.iloc[test_idx[0]]["Year"]),
            "Test_End_Year": int(df.iloc[test_idx[-1]]["Year"]),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5
        })

results = pd.DataFrame(records)
print("\nTime-series cross-validation results:")
print(results.round(4).to_string(index=False))

summary = results.groupby("Model")[["MAE", "RMSE"]].mean().reset_index()
print("\nAverage performance across folds:")
print(summary.round(4).to_string(index=False))

output_file = PROJECT_DIR / "outputs" / "time_series_validation.csv"
results.to_csv(output_file, index=False)
print("\nSaved results to:", output_file)
