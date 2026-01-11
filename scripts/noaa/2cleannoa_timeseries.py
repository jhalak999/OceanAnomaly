import pandas as pd
from pathlib import Path

out_dir = Path("noaa/outputs")

df = pd.read_csv(out_dir / "noaa_oni_features_raw.csv")
df["time"] = pd.to_datetime(df["time"])

df = df.dropna()

df.to_csv(
    out_dir / "noaa_oni_features_clean.csv",
    index=False
)

print("noaa cleaned features saved")
