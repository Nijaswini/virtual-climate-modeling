
from pathlib import Path
import requests
import pandas as pd

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# NASA GISTEMP global monthly temperature dataset
url = (
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/"
    "GLB.Ts+dSST.csv"
)

output_file = DATA_DIR / "global_temperature.csv"

print("Downloading NASA historical temperature data...")

response = requests.get(url, timeout=60)
response.raise_for_status()

output_file.write_bytes(response.content)

# Read the downloaded CSV
df = pd.read_csv(output_file, skiprows=1)

print("\nDownload successful!")
print("Saved to:", output_file)
print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())
