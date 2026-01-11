from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

data_path = BASE_DIR / "dataset/noaa/noaa_oni_features_clean.csv"
out_dir   = BASE_DIR / "dataset/noaa"


# load the data
df = pd.read_csv(data_path)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")


# TIMEBASED SPLIT
train_df = df[df["time"] <= "2015-12-31"]
test_df  = df[df["time"] >= "2016-01-01"]

# saving the splits
train_df.to_csv(out_dir / "oni_train.csv", index=False)
test_df.to_csv(out_dir / "oni_test.csv", index=False)

print("TrainTest split completed")
print(f"Train samples: {len(train_df)}")
print(f"Test samples : {len(test_df)}")
