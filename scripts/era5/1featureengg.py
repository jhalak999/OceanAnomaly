import xarray as xr
import numpy as np


# 1. load merged era5 dataset

ds = xr.open_dataset("era5_ocean_monthly.nc")

print("Original variables:", list(ds.data_vars))


# 2. feature engineering
# 2a) wind speed (m/s)
ds["wind_speed"] = np.sqrt(ds["u10"]**2 + ds["v10"]**2)

# 2b) convert temp from K to C
ds["t2m_c"] = ds["t2m"] - 273.15

# 2c) calc net radiation (J/m²)
ds["net_radiation"] = ds["ssrd"] + ds["strd"]


# 3. final features

features = ds[
    [
        "wind_speed",
        "t2m_c",
        "sshf",
        "net_radiation",
        "e"
    ]
]

print("features:", list(features.data_vars))

# 4. Spatial averaging

features_mean = features.mean(dim=["latitude", "longitude"], skipna=True)

print(features_mean)

# 5. Save processed dataset

features_mean.to_netcdf("era5_features_monthly_timeseries.nc")

print("dataset saved as era5_features_monthly_timeseries.nc")
