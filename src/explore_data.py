
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "global_temperature.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Editorial Climate-Tech palette
CREAM = "#F7F1E7"
PAPER = "#FFFCF7"
INK = "#302B25"
MUTED = "#81776B"
TERRACOTTA = "#C65D3B"
BURNT_ORANGE = "#E68A46"
SAGE = "#71835B"
PALE_SAGE = "#E6EAD9"
PEACH = "#F5D5C2"
GOLD = "#D9AD57"
BORDER = "#E7DACA"

# Consistent chart styling
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
    "axes.labelsize": 10,
    "legend.frameon": False,
    "savefig.facecolor": CREAM,
})

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

# Create polished temperature chart
fig, ax = plt.subplots(figsize=(13, 7))

fig.subplots_adjust(top=0.84, bottom=0.15, left=0.10, right=0.96)

# Highlight recent years without obscuring the data
ax.axvspan(
    max(2000, int(x.min())),
    int(x.max()),
    color=PEACH,
    alpha=0.30,
    zorder=0,
)

# Zero-anomaly reference line
ax.axhline(
    0,
    color=MUTED,
    linewidth=1.2,
    linestyle=":",
    zorder=1,
)

# Observed annual anomalies
ax.plot(
    x,
    y,
    color=TERRACOTTA,
    linewidth=1.8,
    alpha=0.85,
    label="Annual mean anomaly",
    zorder=3,
)

# Linear trend
ax.plot(
    x,
    trend,
    color=INK,
    linewidth=2.6,
    linestyle="--",
    label="Fitted linear trend",
    zorder=4,
)

# Mark the latest available annual observation
latest_idx = np.argmax(x)
ax.scatter(
    x[latest_idx],
    y[latest_idx],
    s=75,
    color=BURNT_ORANGE,
    edgecolor=PAPER,
    linewidth=1.5,
    zorder=5,
)

ax.annotate(
    f"{int(x[latest_idx])}: {y[latest_idx]:.2f} °C",
    xy=(x[latest_idx], y[latest_idx]),
    xytext=(-12, 14),
    textcoords="offset points",
    ha="right",
    color=INK,
    fontsize=9,
    fontweight="bold",
)

# Titles and labels
ax.set_title(
    "Global Temperature Anomalies",
    loc="left",
    pad=22,
)
ax.text(
    0,
    1.035,
    "NASA GISTEMP · Annual mean of available monthly anomalies",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=10,
)

ax.set_xlabel("Year", labelpad=10)
ax.set_ylabel("Temperature anomaly (°C)", labelpad=10)

# Cleaner axis and grid styling
ax.grid(
    axis="y",
    color=BORDER,
    linewidth=0.8,
    alpha=0.8,
)
ax.grid(axis="x", visible=False)

for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

for side in ["left", "bottom"]:
    ax.spines[side].set_color(BORDER)

ax.xaxis.set_major_locator(mdates.YearLocator(20))
# Years are numeric, so use a numeric locator instead.
from matplotlib.ticker import MultipleLocator
ax.xaxis.set_major_locator(MultipleLocator(20))

ax.tick_params(axis="both", length=0, pad=8)

ax.legend(
    loc="upper left",
    ncol=2,
    bbox_to_anchor=(0, 1.005),
)

# Source and interpretation note
fig.text(
    0.10,
    0.045,
    "Source: NASA GISTEMP. Shaded area highlights 2000 onward. "
    "Dashed line is a simple fitted trend, not a climate forecast.",
    fontsize=8.5,
    color=MUTED,
)

plot_file = OUTPUT_DIR / "global_temperature_trend.png"
fig.savefig(plot_file, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

print(
    "\nEstimated linear trend:",
    round(slope * 100, 3),
    "°C per century",
)
print("Graph saved to:", plot_file)
