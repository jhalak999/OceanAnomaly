import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

# -----------------------
# PATHS
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "dataset" / "lstm"
model_dir = BASE_DIR / "results"

# -----------------------
# LOAD DATA
# -----------------------
X_test = np.load(data_dir / "X_test_lstm.npy")
y_test = np.load(data_dir / "y_test_lstm.npy")

# -----------------------
# LOAD MODEL (FIXED)
# -----------------------
model = load_model(model_dir / "lstm_model.h5", compile=False)

# -----------------------
# PREDICT
# -----------------------
y_pred = model.predict(X_test).flatten()

# -----------------------
# PLOT
# -----------------------
plt.figure(figsize=(10,4))
plt.plot(y_test, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted (LSTM)", linestyle="--")

plt.title("LSTM: Actual vs Predicted Ocean Temperature Anomalies")
plt.xlabel("Time Index")
plt.ylabel("Temperature Anomaly")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

