
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

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
rows = []

for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    linear = LinearRegression()
    forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )

    linear.fit(X_train, y.iloc[train_idx])
    forest.fit(X_train, y.iloc[train_idx])

    linear_pred = linear.predict(X_test)
    forest_pred = forest.predict(X_test)

    # Spread between two model predictions
    disagreement = np.abs(linear_pred - forest_pred)

    for i, row_idx in enumerate(test_idx):
        rows.append({
            "Fold": fold,
            "Year": int(df.iloc[row_idx]["Year"]),
            "Actual": float(y.iloc[row_idx]),
            "Linear_Prediction": float(linear_pred[i]),
            "Random_Forest_Prediction": float(forest_pred[i]),
            "Model_Disagreement": float(disagreement[i])
        })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "uncertainty_analysis.csv", index=False)

print("Saved:", OUTPUT_DIR / "uncertainty_analysis.csv")
print("\nModel disagreement summary (°C):")
print(results["Model_Disagreement"].describe().round(4))

print("\nLargest disagreement years:")
print(
    results.nlargest(10, "Model_Disagreement")[
        ["Year", "Actual", "Linear_Prediction",
         "Random_Forest_Prediction", "Model_Disagreement"]
    ].round(4).to_string(index=False)
)
