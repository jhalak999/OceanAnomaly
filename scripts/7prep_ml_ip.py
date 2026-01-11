import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "dataset/final"

# loading merged datasets
train = pd.read_csv(data_dir / "X_train.csv")
test  = pd.read_csv(data_dir / "X_test.csv")


# defining target and features

y_train = train["oni"]
y_test  = test["oni"]

X_train = train.drop(columns=["oni", "valid_time"])
X_test  = test.drop(columns=["oni", "valid_time"])

# saving to csv
X_train.to_csv(data_dir / "X_train_ml.csv", index=False)
X_test.to_csv(data_dir / "X_test_ml.csv", index=False)
y_train.to_csv(data_dir / "y_train.csv", index=False)
y_test.to_csv(data_dir / "y_test.csv", index=False)

print(" ML inputs prepared")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
