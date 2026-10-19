
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSale price statistics:")
print(df["SalePrice"].describe())

print("\nMissing values (top 20):")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\nDuplicate rows:", df.duplicated().sum())

# Sale price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], kde=True)
plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price ($)")
plt.tight_layout()
plt.savefig(REPORTS / "sale_price_distribution.png")
plt.show()

# Living area vs sale price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice")
plt.title("Living Area vs Sale Price")
plt.tight_layout()
plt.savefig(REPORTS / "living_area_vs_price.png")
plt.show()

# Correlation heatmap
cols = [
    "SalePrice",
    "GrLivArea",
    "OverallQual",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
]

plt.figure(figsize=(9, 6))
sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Selected Feature Correlations")
plt.tight_layout()
plt.savefig(REPORTS / "correlation_heatmap.png")
plt.show()
