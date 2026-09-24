import xarray as xr
import numpy as np

file_path = r"C:\Users\Manas Negi\OneDrive\Desktop\SIH Second statement\4572d4b99342c81e5c37ac789460b190\data_stream-oper_stepType-accum.nc"

print("Opening ERA5 dataset...")

ds = xr.open_dataset(file_path)

tp = ds["tp"]

print("\n========== DATA ==========")
print("Time:", ds.valid_time.values[0], "to", ds.valid_time.values[-1])
print("Shape:", tp.shape)
print("Units:", tp.attrs["units"])

# Convert precipitation from meters to millimeters
tp_mm = tp * 1000

print("\nConverted precipitation from meters to millimeters.")

# Create daily totals from hourly precipitation
daily_precip = tp_mm.resample(valid_time="1D").sum()

print("\n========== DAILY DATA ==========")
print("Shape:", daily_precip.shape)
print("First day:", daily_precip.valid_time.values[0])
print("Last day:", daily_precip.valid_time.values[-1])

# Create monthly climatology
monthly_precip = daily_precip.resample(valid_time="1ME").sum()

climatology = monthly_precip.groupby(
    "valid_time.month"
).mean("valid_time")

print("\n========== CLIMATOLOGY ==========")
print(climatology)

# Calculate monthly anomaly
anomaly = monthly_precip.groupby(
    "valid_time.month"
) - climatology

print("\n========== ANOMALY ==========")
print(anomaly)

# Save results
climatology.to_netcdf("precipitation_climatology.nc")
anomaly.to_netcdf("precipitation_anomaly.nc")

print("\nSaved:")
print("  precipitation_climatology.nc")
print("  precipitation_anomaly.nc")