import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
TIME_STEPS = 12

BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "dataset" / "final"
out_dir = BASE_DIR / "dataset" / "lstm"
out_dir.mkdir(exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
X_train = pd.read_csv(data_dir / "X_train.csv")
y_train = pd.read_csv(data_dir / "y_train.csv")

X_test = pd.read_csv(data_dir / "X_test.csv")
y_test = pd.read_csv(data_dir / "y_test.csv")

# Ensure numeric only
X_train = X_train.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

# -----------------------
# SEQUENCE BUILDER
# -----------------------
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:i+time_steps].values)
        ys.append(y.iloc[i+time_steps].values[0])
    return np.array(Xs), np.array(ys)

# -----------------------
# BUILD SEQUENCES
# -----------------------
X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, TIME_STEPS)

# -----------------------
# SAVE
# -----------------------
np.save(out_dir / "X_train_lstm.npy", X_train_seq)
np.save(out_dir / "y_train_lstm.npy", y_train_seq)
np.save(out_dir / "X_test_lstm.npy", X_test_seq)
np.save(out_dir / "y_test_lstm.npy", y_test_seq)

print("✅ LSTM sequences prepared")
print("X_train shape:", X_train_seq.shape)
print("y_train shape:", y_train_seq.shape)
print("X_test shape :", X_test_seq.shape)
print("y_test shape :", y_test_seq.shape)
