
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
# Editorial Climate-Tech theme: no blue shades

CREAM = "#F7F1E7"
PAPER = "#FFFCF7"
INK = "#302B25"
MUTED = "#81776B"
TERRACOTTA = "#C65D3B"
BURNT_ORANGE = "#E68A46"
SAGE = "#71835B"
GOLD = "#D9AD57"
PEACH = "#F5D5C2"
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
fig.subplots_adjust(top=0.83, bottom=0.16, left=0.10, right=0.97)

last_historical_year = int(df["Year"].max())
first_future_year = int(future_years["Year"].min())

# Visually separate the projection period from historical observations
ax.axvspan(
    last_historical_year,
    int(future_years["Year"].max()),
    color=PEACH,
    alpha=0.35,
    zorder=0,
)

# Historical annual anomalies
ax.plot(
    df["Year"],
    y,
    color=TERRACOTTA,
    linewidth=1.8,
    alpha=0.85,
    label="Historical annual anomaly",
    zorder=3,
)

# Fitted historical linear trend
ax.plot(
    df["Year"],
    model.predict(X),
    color=INK,
    linewidth=2.5,
    linestyle="--",
    label="Fitted historical trend",
    zorder=4,
)

# Connect the final historical observation to the projected values
projection_years = pd.concat(
    [
        pd.DataFrame({
            "Year": [last_historical_year],
            "Projected_Anomaly": [
                float(df.loc[df.index[-1], "Annual_Mean_Anomaly"])
            ],
        }),
        future_years,
    ],
    ignore_index=True,
)

ax.plot(
    projection_years["Year"],
    projection_years["Projected_Anomaly"],
    color=BURNT_ORANGE,
    linewidth=2.7,
    linestyle="-",
    marker="o",
    markersize=4.5,
    markerfacecolor=BURNT_ORANGE,
    markeredgecolor=PAPER,
    markeredgewidth=1,
    label="Illustrative linear extrapolation",
    zorder=5,
)

# Projection start marker
ax.axvline(
    last_historical_year,
    color=SAGE,
    linewidth=1.5,
    linestyle=":",
    label=f"Projection starts after {last_historical_year}",
    zorder=2,
)

# Mark the final projected value
final_year = int(future_years["Year"].iloc[-1])
final_value = float(future_years["Projected_Anomaly"].iloc[-1])

ax.annotate(
    f"{final_year}: {final_value:.2f} °C",
    xy=(final_year, final_value),
    xytext=(-8, 12),
    textcoords="offset points",
    ha="right",
    color=INK,
    fontsize=9,
    fontweight="bold",
)

ax.set_title(
    "A Simple Look at Future Temperature Trends",
    loc="left",
    pad=24,
)

ax.text(
    0,
    1.035,
    "Historical observations and a straight-line extrapolation",
    transform=ax.transAxes,
    color=MUTED,
    fontsize=10,
)

ax.set_xlabel("Year", labelpad=10)
ax.set_ylabel("Global temperature anomaly (°C)", labelpad=10)

# Clean grid and borders
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

fig.text(
    0.10,
    0.045,
    "Important: This is a simple statistical extrapolation, not a validated "
    "climate forecast. It does not model emissions scenarios or climate-system dynamics.",
    fontsize=8.2,
    color=MUTED,
)

plot_path = OUTPUT_DIR / "future_projection.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved outputs:")
print(OUTPUT_DIR / "future_projection.csv")
print(plot_path)
