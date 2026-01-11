import xarray as xr

# Lloading era5 features
ds = xr.open_dataset("era5_features_monthly_timeseries.nc")

print("Before cleaning:")
print(ds.isnull().sum())

# it is monthly time data so we will use time interpolation

ds_clean = ds.interpolate_na(dim="valid_time", method="linear")

print("\nAfter cleaning:")
print(ds_clean.isnull().sum())

# save cleaned dataset
ds_clean.to_netcdf("era5_features_clean.nc")

print("\nClean ERA5 dataset saved.")
