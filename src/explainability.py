

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "outputs" / "cleaned_temperature.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load and prepare complete annual observations
df = pd.read_csv(DATA_FILE)

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

for month in months:
    df[month] = pd.to_numeric(df[month], errors="coerce")

df = df[df[months].notna().sum(axis=1) == 12].copy()
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

# Use chronological split: train on earlier years only
split = int(len(df) * 0.8)
X_train = X.iloc[:split]
y_train = y.iloc[:split]
X_test = X.iloc[split:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)

# Explain predictions for the held-out later years
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Mean absolute SHAP value measures average influence magnitude
importance = pd.DataFrame({
    "Feature": features,
    "Mean_Absolute_SHAP": abs(shap_values).mean(axis=0)
}).sort_values("Mean_Absolute_SHAP", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv", index=False
)

print("\nSHAP feature importance:")
print(importance.round(5).to_string(index=False))

# Save feature-importance chart
plt.figure(figsize=(8, 5))
plt.barh(
    importance["Feature"][::-1],
    importance["Mean_Absolute_SHAP"][::-1]
)
plt.xlabel("Mean absolute SHAP value (°C anomaly contribution)")
plt.title("Random Forest Feature Importance using SHAP")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_feature_importance.png", dpi=300)
plt.close()

print("\nSaved:")
print(OUTPUT_DIR / "shap_feature_importance.csv")
print(OUTPUT_DIR / "shap_feature_importance.png")
