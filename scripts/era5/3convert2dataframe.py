import xarray as xr

ds = xr.open_dataset("era5_features_clean.nc")

df = ds.to_dataframe().reset_index()

print(df.head())
print(df.shape)

df.to_csv("era5_features_clean.csv", index=False)

print("CSV saved: era5_features_clean.csv")
