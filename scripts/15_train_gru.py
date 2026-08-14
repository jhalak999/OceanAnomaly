import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "dataset" /  "lstm"


X_train = np.load(data_dir / "X_train_lstm.npy")
X_test  = np.load(data_dir / "X_test_lstm.npy")
y_train = np.load(data_dir / "y_train_lstm.npy")
y_test  = np.load(data_dir / "y_test_lstm.npy")

print("Loaded GRU inputs")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


model = Sequential([
    GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)


y_pred = model.predict(X_test).ravel()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\nGRU Performance")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")
