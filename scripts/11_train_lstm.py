import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# PATHS
# -----------------------
BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "dataset" / "lstm"
out_dir = BASE_DIR / "results"
out_dir.mkdir(exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
X_train = np.load(data_dir / "X_train_lstm.npy")
y_train = np.load(data_dir / "y_train_lstm.npy")
X_test  = np.load(data_dir / "X_test_lstm.npy")
y_test  = np.load(data_dir / "y_test_lstm.npy")

# -----------------------
# MODEL
# -----------------------
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# -----------------------
# TRAIN
# -----------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------
# EVALUATION
# -----------------------
y_pred = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n📌 LSTM Performance")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")

# -----------------------
# SAVE MODEL
# -----------------------
model.save(out_dir / "lstm_model.h5")

# -----------------------
# TRAINING CURVE
# -----------------------
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()
