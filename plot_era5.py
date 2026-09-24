import xarray as xr
import matplotlib.pyplot as plt

file_path = r"C:\Users\Manas Negi\OneDrive\Desktop\SIH Second statement\4572d4b99342c81e5c37ac789460b190\data_stream-oper_stepType-accum.nc"

print("Opening ERA5...")

ds = xr.open_dataset(file_path)

# Select the first hourly timestep
rain = ds["tp"].isel(valid_time=0)

print("Time:", ds.valid_time.values[0])
print("Shape:", rain.shape)
print("Units:", ds["tp"].attrs["units"])

plt.figure(figsize=(10, 7))

rain.plot()

plt.title("ERA5 Total Precipitation")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig("era5_first_map.png", dpi=200)

print("Map saved as era5_first_map.png")