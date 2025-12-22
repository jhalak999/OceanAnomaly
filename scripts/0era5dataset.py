import xarray as xr
ds_acc = xr.open_dataset("dataset/data_stream-moda_stepType-avgad.nc")
ds_ins = xr.open_dataset("dataset/data_stream-moda_stepType-avgua.nc")
print("ACCUMULATED VARIABLES:")
print(list(ds_acc.data_vars))

print("\nINSTANTANEOUS VARIABLES:")
print(list(ds_ins.data_vars))

era5 = xr.merge([ds_acc, ds_ins])
print(era5)

era5.to_netcdf("era5_ocean_monthly.nc")
print("Merged ERA5 dataset saved as era5_ocean_monthly.nc")
