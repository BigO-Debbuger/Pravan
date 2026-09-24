import xarray as xr

file_path = r"C:\Users\Manas Negi\OneDrive\Desktop\SIH Second statement\4572d4b99342c81e5c37ac789460b190\data_stream-oper_stepType-accum.nc"

print("Opening ERA5 dataset...")

ds = xr.open_dataset(file_path)

print("\n========== DATASET ==========")
print(ds)

print("\n========== DIMENSIONS ==========")
print(ds.dims)

print("\n========== COORDINATES ==========")
print(ds.coords)

print("\n========== VARIABLES ==========")
print(ds.data_vars)