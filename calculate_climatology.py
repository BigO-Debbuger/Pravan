import xarray as xr
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent
file_path = ROOT / "era5_tp_india_2010_2024.nc"
monthly_out = ROOT / "precipitation_monthly.nc"
clim_out = ROOT / "precipitation_climatology.nc"
anom_out = ROOT / "precipitation_anomaly.nc"

def main():
    t0 = time.time()
    print("Opening ERA5 dataset...")

    # Open without dask (lazy load)
    ds = xr.open_dataset(file_path)

    print("Calculating monthly precipitation year-by-year (fast and keeps RAM well below 3-4 GB limit)...")
    monthly_data_list = []

    # Xarray explicit chunking: Process year by year to keep RAM usage < 1 GB
    for year in range(2010, 2025):
        # Select year
        ds_year = ds.sel(valid_time=str(year))
        # Load into memory for fast processing (1 year = ~500 MB)
        tp = ds_year["tp"].load()
        
        # Fill NaN (GRIB artefacts) and convert to mm
        tp_mm = tp.fillna(0.0) * 1000.0
        
        # Resample to daily, then monthly
        daily = tp_mm.resample(valid_time="1D").sum(skipna=True)
        monthly = daily.resample(valid_time="1ME").sum(skipna=True)
        
        monthly_data_list.append(monthly)
        print(f"  Processed {year}")

    # Combine all months
    print("Combining all months...")
    monthly_precip = xr.concat(monthly_data_list, dim="valid_time")
    monthly_precip.name = "tp"

    print(f"Saving {monthly_out.name}...")
    monthly_precip.to_netcdf(monthly_out)

    print("Computing 12-month climatology...")
    climatology = monthly_precip.groupby("valid_time.month").mean("valid_time")
    climatology.name = "tp"
    print(f"Saving {clim_out.name}...")
    climatology.to_netcdf(clim_out)

    print("Computing monthly anomaly...")
    anomaly = monthly_precip.groupby("valid_time.month") - climatology
    anomaly.name = "tp"
    print(f"Saving {anom_out.name}...")
    anomaly.to_netcdf(anom_out)

    ds.close()

    t1 = time.time()
    print(f"\nPreprocessing complete in {t1 - t0:.1f} seconds.")

if __name__ == "__main__":
    main()