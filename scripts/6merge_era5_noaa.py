import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

# paths
era5_dir = BASE_DIR / "dataset/era5"
noaa_dir = BASE_DIR / "dataset/noaa"
out_dir  = BASE_DIR / "dataset/final"

out_dir.mkdir(exist_ok=True)


# loading era5 splits

era5_train = pd.read_csv(era5_dir / "train_features_scaled.csv")
era5_test  = pd.read_csv(era5_dir / "test_features_scaled.csv")

era5_train["valid_time"] = pd.to_datetime(era5_train["valid_time"])
era5_test["valid_time"]  = pd.to_datetime(era5_test["valid_time"])


# now loading noaa splits

oni_train = pd.read_csv(noaa_dir / "oni_train.csv")
oni_test  = pd.read_csv(noaa_dir / "oni_test.csv")

oni_train["time"] = pd.to_datetime(oni_train["time"])
oni_test["time"]  = pd.to_datetime(oni_test["time"])


# merging both datasets using InnerJoin on time

train_merged = era5_train.merge(
    oni_train,
    left_on="valid_time",
    right_on="time",
    how="inner"
)

test_merged = era5_test.merge(
    oni_test,
    left_on="valid_time",
    right_on="time",
    how="inner"
)

# dropping the duplicate time column from test and train
train_merged = train_merged.drop(columns=["time"])
test_merged  = test_merged.drop(columns=["time"])


# noe save the final datasets

train_merged.to_csv(out_dir / "X_train.csv", index=False)
test_merged.to_csv(out_dir / "X_test.csv", index=False)

print("ERA5 and NOAA are merged.")
print(f"Train shape: {train_merged.shape}")
print(f"Test shape : {test_merged.shape}")
print("Command done")
