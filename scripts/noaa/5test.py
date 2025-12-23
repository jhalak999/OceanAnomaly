import pandas as pd
from pathlib import Path

# Paths
data_path = Path("noaa/outputs/noaa_oni_features_clean.csv")
out_dir = Path("noaa/outputs")

# Load data
df = pd.read_csv(data_path)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

# -------------------------------
# TIME-BASED SPLIT
# -------------------------------
train_df = df[df["time"] <= "2015-12-31"]
test_df  = df[df["time"] >= "2016-01-01"]

# Save splits
train_df.to_csv(out_dir / "oni_train.csv", index=False)
test_df.to_csv(out_dir / "oni_test.csv", index=False)

print("✅ Train–Test split completed")
print(f"Train samples: {len(train_df)}")
print(f"Test samples : {len(test_df)}")
