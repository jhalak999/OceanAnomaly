import pandas as pd

df = pd.read_csv("era5_features_clean.csv")

split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

train.to_csv("train_features_raw.csv", index=False)
test.to_csv("test_features_raw.csv", index=False)

print("Train:", train.shape)
print("Test:", test.shape)
