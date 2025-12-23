import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load raw splits
train = pd.read_csv("train_features_raw.csv")
test = pd.read_csv("test_features_raw.csv")

# Separate time
train_time = train["valid_time"]
test_time = test["valid_time"]

X_train = train.drop(columns=["valid_time"])
X_test = test.drop(columns=["valid_time"])

# Fit ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Back to DataFrame
train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

train_scaled["valid_time"] = train_time
test_scaled["valid_time"] = test_time

# Save
train_scaled.to_csv("train_features_scaled.csv", index=False)
test_scaled.to_csv("test_features_scaled.csv", index=False)

print("Scaling done correctly (no data leakage)")
