
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "outputs" / "cleaned_temperature.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATA_FILE)

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# Ensure month values are numeric
for month in months:
    df[month] = pd.to_numeric(df[month], errors="coerce")

# For model training, use only years with all 12 months available
df["Valid_Months"] = df[months].notna().sum(axis=1)
df = df[df["Valid_Months"] == 12].copy()

# Recalculate annual means from complete years only
df["Annual_Mean_Anomaly"] = df[months].mean(axis=1)
df = df.sort_values("Year").reset_index(drop=True)

# Create historical features to predict the following year's anomaly
df["Lag_1"] = df["Annual_Mean_Anomaly"].shift(1)
df["Lag_2"] = df["Annual_Mean_Anomaly"].shift(2)
df["Rolling_3yr_Mean"] = (
    df["Annual_Mean_Anomaly"].shift(1).rolling(3).mean()
)

df = df.dropna().reset_index(drop=True)

features = ["Year", "Lag_1", "Lag_2", "Rolling_3yr_Mean"]
X = df[features]
y = df["Annual_Mean_Anomaly"]

# Preserve chronological order: earlier years train, later years test
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Training years:", int(df.iloc[0]["Year"]), "to",
      int(df.iloc[split - 1]["Year"]))
print("Testing years:", int(df.iloc[split]["Year"]), "to",
      int(df.iloc[-1]["Year"]))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=3,
        random_state=42
    )
}

results = []
predictions = {}

# Baseline: predict next year's anomaly using the previous year
baseline_pred = X_test["Lag_1"].to_numpy()
predictions["Previous-year baseline"] = baseline_pred

results.append({
    "Model": "Previous-year baseline",
    "MAE": mean_absolute_error(y_test, baseline_pred),
    "RMSE": mean_squared_error(y_test, baseline_pred) ** 0.5,
    "R2": r2_score(y_test, baseline_pred)
})

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\nModel comparison:")
print(results_df.round(4).to_string(index=False))

# Save predictions for the test period
prediction_df = pd.DataFrame({
    "Year": df["Year"].iloc[split:].to_numpy(),
    "Actual": y_test.to_numpy(),
    "Linear_Regression": predictions["Linear Regression"],
    "Random_Forest": predictions["Random Forest"]
})
prediction_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)


# Plot observed versus predicted values — Editorial Climate-Tech theme

CREAM = "#F7F1E7"
PAPER = "#FFFCF7"
INK = "#302B25"
MUTED = "#81776B"
TERRACOTTA = "#C65D3B"
BURNT_ORANGE = "#E68A46"
SAGE = "#71835B"
GOLD = "#D9AD57"
BORDER = "#E7DACA"

plt.rcParams.update({
    "figure.facecolor": CREAM,
    "axes.facecolor": PAPER,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "savefig.facecolor": CREAM,
})

fig, ax = plt.subplots(figsize=(13, 7))
fig.subplots_adjust(top=0.82, bottom=0.15, left=0.10, right=0.97)

years = prediction_df["Year"]

# Highlight the gap between observed values and predictions
ax.fill_between(
    years,
    prediction_df["Actual"],
    prediction_df["Linear_Regression"],
    color=PEACH if "PEACH" in globals() else "#F5D5C2",
    alpha=0.28,
    label="_nolegend_",
    zorder=1,
)

# Observed data
ax.plot(
    years,
    prediction_df["Actual"],
    color=INK,
    linewidth=2.7,
    marker="o",
    markersize=5.5,
    markerfacecolor=BURNT_ORANGE,
    markeredgecolor=PAPER,
    markeredgewidth=1.2,
    label="Observed",
    zorder=5,
)

# Linear regression
ax.plot(
    years,
    prediction_df["Linear_Regression"],
    color=TERRACOTTA,
    linewidth=2.4,
    marker="s",
    markersize=4,
    label="Linear Regression",
    zorder=4,
)

# Random forest
ax.plot(
    years,
    prediction_df["Random_Forest"],
    color=SAGE,
    linewidth=2.4,
    marker="^",
    markersize=4.5,
    label="Random Forest",
    zorder=3,
)

# Previous-year baseline
ax.plot(
    years,
    X_test["Lag_1"].to_numpy(),
    color=GOLD,
    linewidth=2.2,
    linestyle="--",
    label="Previous-year baseline",
    zorder=2,
)

# Titles and supporting description
ax.set_title(
    "How Well Do the Models Track Temperature?",
    loc="left",
    pad=24,
)

ax.text(
    0,
    1.035,
    "Held-out test period · Chronological split · Temperature anomaly (°C)",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=10,
)

ax.set_xlabel("Year", labelpad=10)
ax.set_ylabel("Temperature anomaly (°C)", labelpad=10)

# Subtle grid and clean borders
ax.grid(axis="y", color=BORDER, linewidth=0.8)
ax.grid(axis="x", visible=False)

for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

for side in ["left", "bottom"]:
    ax.spines[side].set_color(BORDER)

ax.tick_params(axis="both", length=0, pad=8)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0, 1.005),
    ncol=2,
)

# Add a clear scientific interpretation
fig.text(
    0.10,
    0.045,
    "Note: Predictions are evaluated on later years not used for training. "
    "Visual agreement does not establish forecasting skill.",
    fontsize=8.5,
    color=MUTED,
)

plot_path = OUTPUT_DIR / "model_predictions.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

print("\nSaved model comparison:", OUTPUT_DIR / "model_comparison.csv")
print("Saved test predictions:", OUTPUT_DIR / "test_predictions.csv")
print("Saved prediction graph:", plot_path)
