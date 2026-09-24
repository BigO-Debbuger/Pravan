import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

ANOMALY_FILE = Path(__file__).resolve().parent / "precipitation_anomaly.nc"

# Load precomputed anomaly directly instead of recomputing from raw ERA5
ds = xr.open_dataset(ANOMALY_FILE)
anomaly = ds["tp"]

# Pick the last available month (Sep 2022)
selected = anomaly.isel(valid_time=-1)

date_str = str(selected.valid_time.values)[:10]
print(f"Anomaly date: {date_str}")
print(f"Minimum anomaly: {float(selected.min()):.2f} mm")
print(f"Maximum anomaly: {float(selected.max()):.2f} mm")

plt.figure(figsize=(11, 7))

selected.plot(
    cmap="RdBu_r",
    center=0,
    cbar_kwargs={"label": "Precipitation Anomaly (mm)"}
)

plt.title(f"ERA5 Monthly Precipitation Anomaly - {date_str}")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")

plt.tight_layout()
output_path = Path(__file__).resolve().parent / "era5_anomaly_map.png"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Saved: {output_path.name}")