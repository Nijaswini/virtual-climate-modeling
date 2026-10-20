

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


# Save feature-importance chart — Editorial Climate-Tech theme

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
    "savefig.facecolor": CREAM,
})

# Highest importance at the top
plot_data = importance.sort_values(
    "Mean_Absolute_SHAP", ascending=True
).reset_index(drop=True)

# Use a warm gradient based on feature importance
max_importance = plot_data["Mean_Absolute_SHAP"].max()
bar_colors = []

for value in plot_data["Mean_Absolute_SHAP"]:
    if max_importance > 0 and value >= max_importance * 0.65:
        bar_colors.append(TERRACOTTA)
    elif max_importance > 0 and value >= max_importance * 0.25:
        bar_colors.append(BURNT_ORANGE)
    else:
        bar_colors.append(SAGE)

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.subplots_adjust(top=0.80, bottom=0.17, left=0.25, right=0.96)

bars = ax.barh(
    plot_data["Feature"],
    plot_data["Mean_Absolute_SHAP"],
    color=bar_colors,
    height=0.58,
    edgecolor=PAPER,
    linewidth=1.2,
)

# Add readable values at the end of each bar
max_value = max_importance if pd.notna(max_importance) else 0
label_offset = max_value * 0.025 if max_value > 0 else 0.001

for bar, value in zip(bars, plot_data["Mean_Absolute_SHAP"]):
    ax.text(
        bar.get_width() + label_offset,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center",
        ha="left",
        fontsize=9,
        color=INK,
    )

ax.set_title(
    "What Influences the Random Forest Model?",
    loc="left",
    pad=24,
)

ax.text(
    0,
    1.035,
    "SHAP feature importance · Held-out later years",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=10,
)

ax.set_xlabel(
    "Mean absolute SHAP value (°C anomaly contribution)",
    labelpad=10,
)
ax.set_ylabel("Input feature", labelpad=10)

ax.grid(axis="x", color=BORDER, linewidth=0.8)
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)

for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)

ax.spines["bottom"].set_color(BORDER)
ax.tick_params(axis="both", length=0, pad=8)

# Leave room for value labels
ax.set_xlim(0, max_value * 1.22 if max_value > 0 else 1)

fig.text(
    0.10,
    0.045,
    "Interpretation: SHAP describes how features influence this fitted model; "
    "it does not establish causation.",
    fontsize=8.5,
    color=MUTED,
)

plot_path = OUTPUT_DIR / "shap_feature_importance.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(OUTPUT_DIR / "shap_feature_importance.csv")
print(plot_path)

