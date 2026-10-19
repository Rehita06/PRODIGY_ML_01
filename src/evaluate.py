from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
y = df["SalePrice"]

# Must match the split used in train.py.
_, X_valid, _, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load(ROOT / "models" / "house_price_model.joblib")
predictions = model.predict(X_valid)

mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))
r2 = r2_score(y_valid, predictions)

print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:    {r2:.4f}")

# Actual vs predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_valid, predictions, alpha=0.6)

low = min(y_valid.min(), predictions.min())
high = max(y_valid.max(), predictions.max())

plt.plot([low, high], [low, high], "r--")
plt.xlabel("Actual Sale Price ($)")
plt.ylabel("Predicted Sale Price ($)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig(ROOT / "reports" / "actual_vs_predicted.png")
plt.show()

# Residuals
residuals = y_valid - predictions

plt.figure(figsize=(7, 5))
plt.scatter(predictions, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Sale Price ($)")
plt.ylabel("Residual: Actual - Predicted ($)")
plt.title("Prediction Residuals")
plt.tight_layout()
plt.savefig(ROOT / "reports" / "residuals.png")
plt.show()