from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"

model = joblib.load(MODEL_PATH)

# Use a real row from the training dataset as a template so
# all expected feature columns are present.
df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")

sample_house = df.drop(
    columns=["SalePrice", "Id"], errors="ignore"
).iloc[[0]].copy()

predicted_price = model.predict(sample_house)[0]

print("Example property:")
print(sample_house.T)
print(f"\nPredicted sale price: ${predicted_price:,.2f}")