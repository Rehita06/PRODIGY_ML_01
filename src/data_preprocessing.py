
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "OverallQual",
    "GarageCars",
    "YearBuilt",
]

target = "SalePrice"

data = df[features + [target]].copy()

print("Missing values before cleaning:")
print(data.isnull().sum())

# Fill missing feature values using their medians
for column in features:
    data[column] = data[column].fillna(data[column].median())

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nPrepared dataset shape:", data.shape)
print("\nTarget statistics:")
print(data[target].describe())

output_path = OUTPUT_DIR / "house_prices_clean.csv"
data.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to: {output_path}")
