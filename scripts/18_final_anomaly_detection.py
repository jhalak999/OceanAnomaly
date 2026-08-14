import pandas as pd
import joblib
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "dataset" / "final"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_PATH)
print("Model loaded")

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")

print("Test data loaded")
print("X_test shape:", X_test.shape)

if "valid_time" in X_test.columns:
    X_test = X_test.drop(columns=["valid_time"])


if "oni" in X_test.columns:
    X_test = X_test.drop(columns=["oni"])

print("After cleaning shape:", X_test.shape)

y_pred = model.predict(X_test)

def classify(val):
    if val >= 0.5:
        return "El Nino"
    elif val <= -0.5:
        return "La Nina"
    else:
        return "Neutral"

pred_class = [classify(v) for v in y_pred]
actual_class = [classify(v) for v in y_test.values.flatten()]

out = pd.DataFrame({
    "Actual_ONI": y_test.values.flatten(),
    "Predicted_ONI": y_pred,
    "Actual_Class": actual_class,
    "Predicted_Class": pred_class
})

out_path = RESULTS_DIR / "final_anomaly_predictions.csv"
out.to_csv(out_path, index=False)

print("\nFINAL RESULTS SAVED:", out_path)


print("\nSample Predictions:\n")
print(out.head(15))

import matplotlib.pyplot as plt
import numpy as np

print("\nGenerating ENSO classification plot...")


time_axis = np.arange(len(y_pred))

plt.figure(figsize=(12,6))


plt.plot(time_axis, y_pred, label="Predicted ONI")


plt.axhline(0.5)
plt.axhline(-0.5)

plt.fill_between(time_axis, 0.5, max(y_pred)+0.5, alpha=0.1)
plt.fill_between(time_axis, min(y_pred)-0.5, -0.5, alpha=0.1)

plt.xlabel("Time Index")
plt.ylabel("ONI Value")
plt.title("ENSO Phase Classification Based on Predicted ONI")
plt.legend()


plot_path = RESULTS_DIR / "enso_classification_timeline.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print("ENSO classification plot saved at:", plot_path)

plt.close()