"""
ml/preprocessing.py
-------------------
Loads and preprocesses ERA5 precipitation data for ML feature engineering.

Inputs:
  - Raw ERA5 NetCDF  (hourly accumulated precipitation, metres)
  - precipitation_climatology.nc  (monthly climatology baseline, mm)
  - precipitation_anomaly.nc      (monthly anomaly, mm)

Outputs (returned as xarray DataArrays / DataFrames):
  - daily_precip   : (days, lat, lon) daily rainfall totals in mm, NaN-filled
  - monthly_precip : (months, lat, lon) monthly rainfall totals in mm
  - climatology    : (12, lat, lon) mean monthly baseline in mm
  - anomaly        : (months, lat, lon) anomaly = monthly - climatology in mm

Design decisions:
  - NaN values in daily_precip arise from GRIB accumulation step artefacts.
    They are filled with 0.0 because the variable is total precipitation
    accumulation; missing accumulation hours contribute zero rainfall.
  - All transformations are computed on the FULL dataset here.
    Train/test splitting and per-split scaling happen in features.py
    to avoid leakage.
  - We do NOT reload the 169 MB raw file unless daily_precip is explicitly
    requested (load_daily=False by default) to keep fast iterations possible.
"""

import pathlib
import numpy as np
import xarray as xr
import pandas as pd

# ---- paths -----------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent   # project root
RAW_NC = ROOT / "era5_tp_india_2010_2024.nc"
CLIM_NC = ROOT / "precipitation_climatology.nc"
ANOM_NC = ROOT / "precipitation_anomaly.nc"


# ---- loaders ----------------------------------------------------------------

def load_climatology() -> xr.DataArray:
    """Load monthly climatology baseline (12 × lat × lon), mm."""
    ds = xr.open_dataset(CLIM_NC)
    return ds["tp"]          # shape: (12, 125, 121)


def load_anomaly() -> xr.DataArray:
    """Load monthly precipitation anomaly (28 months × lat × lon), mm."""
    ds = xr.open_dataset(ANOM_NC)
    return ds["tp"]          # shape: (28, 125, 121), dim=valid_time


def load_daily(fill_nan: bool = True) -> xr.DataArray:
    """
    Load raw ERA5, convert to mm, resample to daily totals.

    NaN handling:
      GRIB hourly accumulations sometimes produce NaN at step boundaries.
      These are filled with 0.0 (no rainfall contribution) before summing.
      fill_nan=True enables this; set False only for diagnostic purposes.

    Returns: DataArray (days × lat × lon), mm/day.
    """
    print("[preprocessing] Loading raw ERA5 NetCDF with Dask chunking...")
    ds = xr.open_dataset(RAW_NC, chunks={"valid_time": 720})
    tp_mm = ds["tp"] * 1000.0   # m -> mm

    if fill_nan:
        tp_mm = tp_mm.fillna(0.0)

    daily = tp_mm.resample(valid_time="1D").sum(skipna=True)
    print(f"[preprocessing] Daily data shape: {daily.shape}  "
          f"({len(daily.valid_time)} days)")
    return daily                # shape: (852, 125, 121)


def load_monthly_from_daily(daily: xr.DataArray) -> xr.DataArray:
    """Aggregate daily totals to monthly totals. Returns (months × lat × lon)."""
    return daily.resample(valid_time="1ME").sum(skipna=True)


def load_all(load_daily: bool = False) -> dict:
    """
    Convenience loader.

    Returns a dict with:
      'climatology'     : xr.DataArray (12, lat, lon)
      'anomaly'         : xr.DataArray (months, lat, lon)
      'monthly_precip'  : xr.DataArray (months, lat, lon)  -- only if load_daily=True
      'daily_precip'    : xr.DataArray (days, lat, lon)    -- only if load_daily=True
    """
    result = {
        "climatology": load_climatology(),
        "anomaly":     load_anomaly(),
    }

    if load_daily:
        daily = load_daily()
        result["daily_precip"]   = daily
        result["monthly_precip"] = load_monthly_from_daily(daily)

    return result


# ---- quick self-test --------------------------------------------------------

if __name__ == "__main__":
    print("=== Testing preprocessing.py ===\n")

    clim = load_climatology()
    print(f"Climatology shape : {clim.shape}")
    print(f"  months          : {clim.month.values}")
    print(f"  lat range       : {float(clim.latitude.min()):.2f} - {float(clim.latitude.max()):.2f}")
    print(f"  lon range       : {float(clim.longitude.min()):.2f} - {float(clim.longitude.max()):.2f}")
    print(f"  NaN count       : {int(clim.isnull().sum())}")

    anom = load_anomaly()
    print(f"\nAnomaly shape     : {anom.shape}")
    print(f"  time range      : {str(anom.valid_time.values[0])[:10]} "
          f"to {str(anom.valid_time.values[-1])[:10]}")
    print(f"  NaN count       : {int(anom.isnull().sum())}")
    print(f"  global mean     : {float(anom.mean()):.6f} mm  (should be ~0)")
    print(f"  global std      : {float(anom.std()):.2f} mm")
    print(f"  min / max       : {float(anom.min()):.2f} / {float(anom.max()):.2f} mm")

    print("\n[preprocessing.py] Self-test PASSED.")
