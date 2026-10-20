
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "outputs" / "cleaned_temperature.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data and retain years with all 12 monthly observations
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

X = df[["Year"]]
y = df["Annual_Mean_Anomaly"]

# Fit a simple historical linear trend
model = LinearRegression()
model.fit(X, y)

# Project the next 10 years
future_years = pd.DataFrame({
    "Year": range(int(df["Year"].max()) + 1,
                  int(df["Year"].max()) + 11)
})

future_years["Projected_Anomaly"] = model.predict(
    future_years[["Year"]]
)

future_years.to_csv(
    OUTPUT_DIR / "future_projection.csv", index=False
)

print("Historical data years:",
      int(df["Year"].min()), "to", int(df["Year"].max()))
print("Historical trend slope (°C per year):",
      round(model.coef_[0], 5))
print("\nIllustrative projections:")
print(future_years.round(4).to_string(index=False))

# Plot historical anomalies, fitted trend, and extrapolation
plt.figure(figsize=(11, 6))
plt.plot(
    df["Year"], y,
    label="Historical annual anomaly"
)
plt.plot(
    df["Year"], model.predict(X),
    label="Fitted linear trend"
)
plt.plot(
    future_years["Year"],
    future_years["Projected_Anomaly"],
    linestyle="--",
    marker="o",
    label="Illustrative projection"
)
plt.axvline(
    df["Year"].max(),
    linestyle=":",
    label="Projection starts"
)
plt.xlabel("Year")
plt.ylabel("Global temperature anomaly (°C)")
plt.title("Historical Temperature Anomalies and Linear-Trend Projection")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "future_projection.png",
    dpi=300
)
plt.close()

print("\nSaved outputs:")
print(OUTPUT_DIR / "future_projection.csv")
print(OUTPUT_DIR / "future_projection.png")
