import xarray as xr
import pandas as pd
from pathlib import Path

out_dir = Path("noaa/outputs")

ds = xr.open_dataset(out_dir / "noaa_oni_monthly.nc")
df = ds.to_dataframe().reset_index()

df["oni_3m_mean"] = df["oni"].rolling(3).mean()
df["oni_lag1"] = df["oni"].shift(1)
df["oni_lag3"] = df["oni"].shift(3)

df.to_csv(
    out_dir / "noaa_oni_features_raw.csv",
    index=False
)

print("noaa feature engineering completed")
