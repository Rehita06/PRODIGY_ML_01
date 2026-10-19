
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Use the same six features as app.py
features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "OverallQual",
    "GarageCars",
    "YearBuilt",
]

target = "SalePrice"

X = df[features].copy()
y = df[target].copy()

# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Create a pipeline to handle missing values and train the model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", LinearRegression()),
])

# Train model
print("Training Linear Regression model...")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:   {r2:.4f}")

# Save model using the exact filename expected by app.py
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
joblib.dump(model, MODEL_PATH)

# Save evaluation metrics
metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2],
})

METRICS_PATH = REPORTS_DIR / "linear_regression_metrics.csv"
metrics.to_csv(METRICS_PATH, index=False)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")
print("\nTraining completed successfully!")
