
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "global_temperature.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE, skiprows=1)


# Monthly temperature anomaly columns
months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# Convert invalid values such as *** into missing values
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

for month in months:
    df[month] = pd.to_numeric(df[month], errors="coerce")

# Report data quality
print("Original dataset shape:", df.shape)
print("\nMissing values in monthly columns:")
print(df[months].isna().sum())

# Calculate annual mean from available monthly values
df["Annual_Mean_Anomaly"] = df[months].mean(axis=1)

# Keep rows with a valid year and at least one monthly value
clean_df = df.dropna(
    subset=["Year", "Annual_Mean_Anomaly"]
).copy()

clean_df["Year"] = clean_df["Year"].astype(int)

# Save cleaned data
clean_file = OUTPUT_DIR / "cleaned_temperature.csv"
clean_df.to_csv(clean_file, index=False)

print("\nCleaned dataset shape:", clean_df.shape)
print("Years covered:", clean_df["Year"].min(), "-", clean_df["Year"].max())
print("Cleaned data saved to:", clean_file)

print("\nFirst five annual means:")
print(clean_df[["Year", "Annual_Mean_Anomaly"]].head())

# Fit a simple linear trend for visualization
x = clean_df["Year"].to_numpy()
y = clean_df["Annual_Mean_Anomaly"].to_numpy()
slope, intercept = np.polyfit(x, y, 1)
trend = slope * x + intercept

# Plot observed anomalies and fitted trend
plt.figure(figsize=(12, 6))
plt.plot(x, y, label="Annual mean temperature anomaly")
plt.plot(x, trend, "--", label="Linear trend")
plt.axhline(0, linewidth=1)
plt.title("Global Temperature Anomaly: NASA GISTEMP")
plt.xlabel("Year")
plt.ylabel("Temperature anomaly (°C)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_file = OUTPUT_DIR / "global_temperature_trend.png"
plt.savefig(plot_file, dpi=150)
plt.show()

print("\nEstimated linear trend:", round(slope * 100, 3),
      "°C per century")
print("Graph saved to:", plot_file)
