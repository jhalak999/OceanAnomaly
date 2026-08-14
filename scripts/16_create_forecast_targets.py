import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset" / "final"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
print("Loaded XGBoost model")


X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

DROP_COLS = ["valid_time", "oni"]  

X_test = X_test.drop(columns=[c for c in DROP_COLS if c in X_test.columns])

print("After cleanup X_test shape:", X_test.shape)


y_pred = model.predict(X_test)


pred_df = pd.DataFrame({
    "actual_anomaly": y_test.values.flatten(),
    "predicted_anomaly": y_pred,
    "residual": y_test.values.flatten() - y_pred
})

pred_path = RESULTS_DIR / "xgboost_predictions.csv"
pred_df.to_csv(pred_path, index=False)

print(f"Predictions saved to {pred_path}")

plt.figure(figsize=(10, 4))
plt.plot(pred_df["actual_anomaly"], label="Actual", linewidth=2)
plt.plot(pred_df["predicted_anomaly"], label="Predicted", linestyle="--")
plt.title("XGBoost: Actual vs Predicted Ocean Temperature Anomalies")
plt.xlabel("Time Index")
plt.ylabel("ONI Anomaly")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
