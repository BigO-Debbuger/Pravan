"""
ml/features.py
--------------
Builds the ML-ready feature matrix and target vector from ERA5 precipitation data.

PREDICTION PROBLEM DESIGN
==========================
Target:
  Continuous monthly precipitation anomaly (mm) at each grid cell.
  Rationale: With only 28 monthly time steps and 15,125 spatial grid cells,
  regression on a continuous target preserves the most information.
  Categorical (e.g., wet/dry) would discard magnitude and require
  an arbitrary threshold that is not defensible with <3 years of data.
  A future categorical risk layer can be derived POST-prediction from the
  continuous output (e.g., anomaly > +1 sigma = 'excess', < -1 sigma = 'deficit').

Features constructed (all from ERA5 monthly data to avoid leakage):
  --- Temporal ---
  month             : calendar month (1-12), captures seasonality
  month_sin         : sin(2π·month/12) — cyclic encoding
  month_cos         : cos(2π·month/12) — cyclic encoding
  year              : calendar year (trend proxy)
  months_since_start: integer index (0, 1, 2, ...) for linear trend
  --- Spatial ---
  lat               : latitude of grid cell
  lon               : longitude of grid cell
  lat_norm          : normalised latitude (z-score)
  lon_norm          : normalised longitude (z-score)
  --- Climatology ---
  climatology_mm    : long-term mean for that calendar month at that cell
  --- Lag anomaly features (no future leakage — only past months) ---
  anomaly_lag1      : anomaly 1 month prior  (t-1)
  anomaly_lag2      : anomaly 2 months prior (t-2)
  anomaly_lag3      : anomaly 3 months prior (t-3)
  --- Lag rainfall features ---
  precip_lag1       : total rainfall (mm) 1 month prior
  precip_lag2       : total rainfall (mm) 2 months prior
  rolling_mean_3m   : 3-month rolling mean of monthly rainfall (months t-3..t-1)
  rolling_std_3m    : 3-month rolling std  of monthly rainfall (months t-3..t-1)
  rolling_mean_6m   : 6-month rolling mean of monthly rainfall (months t-6..t-1)

Target:
  anomaly_mm        : precipitation anomaly at current month t (mm)

DATA LEAKAGE PREVENTION
=======================
  - All lag/rolling features are strictly lookback-only (t-1 and earlier).
  - The current month's rainfall (monthly_precip at t) is EXCLUDED from features.
  - Train/val/test split is chronological (no shuffle).
  - StandardScaler is fit ONLY on training rows and applied to val/test.

SPLIT STRATEGY
==============
  With 180 months of data (2010-2024):
    Train : months 6..131   (Jul 2010 – Dec 2020, 126 months)
    Val   : months 132..155 (Jan 2021 – Dec 2022, 24 months)
    Test  : months 156..179 (Jan 2023 – Dec 2024, 24 months)

  NOTE: Due to lag features, rows where lag windows are undefined
  (the first 6 months: Jan-Jun 2010) are dropped from the dataset.
  These rows have NaN lag features and cannot be used for training.

OUTPUT
======
  ml/processed/features_train.parquet
  ml/processed/features_val.parquet
  ml/processed/features_test.parquet
  ml/processed/feature_names.txt
  ml/processed/split_summary.txt
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import xarray as xr

# Allow running directly from project root or from ml/
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from preprocessing import load_climatology, load_anomaly

OUT_DIR = ROOT / "ml" / "processed"

# ---- Split configuration ---------------------------------------------------
N_MONTHS      = 180  # total months in anomaly file (2010-2024)
VAL_START_IDX = 132  # index of first validation month (Jan 2021)
TEST_START_IDX = 156 # index of first test month (Jan 2023)

# ---- helpers ----------------------------------------------------------------

def _cyclic_encode(series: pd.Series, period: int):
    """Return (sin, cos) cyclic encoding for a periodic integer feature."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


# ---- main pipeline ----------------------------------------------------------

def build_feature_matrix(verbose: bool = True) -> dict:
    """
    Builds and returns a dict containing:
      'df_train', 'df_val', 'df_test'    : DataFrames with features + target
      'feature_cols'                      : list of feature column names
      'target_col'                        : name of target column ('anomaly_mm')
      'scaler_params'                     : dict with mean/std for each feature
                                            (fit on train only)
    """
    if verbose:
        print("=" * 60)
        print("  ERA5 ML FEATURE ENGINEERING PIPELINE")
        print("=" * 60)

    # ---- 1. Load derived data -----------------------------------------------
    if verbose:
        print("\n[1/6] Loading climatology and anomaly...")

    clim = load_climatology()   # (12, lat, lon)
    anom = load_anomaly()       # (28, lat, lon)

    # Also need monthly totals to build lag rainfall features.
    # These are NOT in the saved files, so reconstruct from anomaly + climatology:
    #   monthly_total(t) = anomaly(t) + climatology(month_of_t)
    # This is exact because anomaly was defined as: anomaly = monthly - climatology
    # and carries NO leakage (climatology was computed on the full dataset,
    # but for lag features we only use past months, so the climatology itself
    # is the seasonal mean pattern, not a future observation).
    months_of_anomaly = anom.valid_time.dt.month.values   # shape: (28,)
    clim_aligned = clim.sel(month=months_of_anomaly)       # broadcast-friendly
    # monthly_total(t) = anom(t) + clim(month_of_t)
    monthly_total = anom + clim_aligned.values             # (28, lat, lon)

    lats  = clim.latitude.values    # shape: (125,)
    lons  = clim.longitude.values   # shape: (121,)
    times = pd.DatetimeIndex(anom.valid_time.values)       # 28 timestamps

    n_lat = len(lats)
    n_lon = len(lons)

    if verbose:
        print(f"   Anomaly  : {anom.shape}  ({len(times)} months)")
        print(f"   Clim     : {clim.shape}  (12 calendar months)")
        print(f"   Grid     : {n_lat} lat × {n_lon} lon = {n_lat*n_lon:,} cells")

    # ---- 2. Convert to 2D arrays for vectorised operations ------------------
    if verbose:
        print("\n[2/6] Extracting numpy arrays...")

    anom_arr  = anom.values           # (28, 125, 121)
    clim_arr  = clim.values           # (12, 125, 121)
    total_arr = monthly_total.values  # (28, 125, 121)

    # ---- 3. Build lag matrices at spatial level ------------------------------
    # For each time index t and each grid cell (i,j):
    #   anomaly_lag1[t]  = anom_arr[t-1]   (NaN if t-1 < 0)
    #   precip_lag1[t]   = total_arr[t-1]
    #   rolling_mean_3m[t] = mean(total_arr[t-3:t-1]) strictly lookback

    if verbose:
        print("\n[3/6] Building lag and rolling features...")

    def lag(arr, k):
        """Shift arr along axis-0 by k steps (fill leading k with NaN)."""
        out = np.empty_like(arr, dtype=np.float32)
        out[:k] = np.nan
        out[k:] = arr[:-k] if k > 0 else arr
        return out

    def rolling_mean(arr, window):
        """
        Strict lookback rolling mean:
          result[t] = mean(arr[t-window : t])
        Leading 'window' entries are NaN.
        """
        T, H, W = arr.shape
        out = np.full_like(arr, np.nan, dtype=np.float32)
        for t in range(window, T):
            out[t] = arr[t - window:t].mean(axis=0)
        return out

    def rolling_std(arr, window):
        """Strict lookback rolling std (ddof=1); NaN for t < window."""
        T, H, W = arr.shape
        out = np.full_like(arr, np.nan, dtype=np.float32)
        for t in range(window, T):
            with np.errstate(invalid='ignore'):
                out[t] = arr[t - window:t].std(axis=0, ddof=1)
        return out

    # Lag anomaly
    anom_lag1 = lag(anom_arr,  1)
    anom_lag2 = lag(anom_arr,  2)
    anom_lag3 = lag(anom_arr,  3)

    # Lag rainfall totals
    precip_lag1 = lag(total_arr, 1)
    precip_lag2 = lag(total_arr, 2)

    # Rolling rainfall features
    roll_mean3 = rolling_mean(total_arr, 3)   # needs t-3..t-1, valid from t=3
    roll_std3  = rolling_std(total_arr,  3)
    roll_mean6 = rolling_mean(total_arr, 6)   # needs t-6..t-1, valid from t=6

    # ---- 4. Flatten to per-sample rows --------------------------------------
    if verbose:
        print("\n[4/6] Flattening grid to row-per-sample...")

    # Coordinate grids
    lat_grid  = np.tile(lats[:, None], (1, n_lon))   # (125, 121)
    lon_grid  = np.tile(lons[None, :], (n_lat, 1))   # (125, 121)

    # Normalise lat/lon (fit on full dataset — these are static coordinates,
    # not observations, so no leakage risk)
    lat_mean, lat_std = lats.mean(), lats.std()
    lon_mean, lon_std = lons.mean(), lons.std()

    rows = []
    for t_idx, timestamp in enumerate(times):
        m   = timestamp.month   # 1-12
        yr  = timestamp.year
        t0  = t_idx             # months_since_start

        # Scalar features per time step (broadcast to all cells)
        sin_m, cos_m = _cyclic_encode(pd.Series([m]), 12)
        sin_m, cos_m = float(sin_m.iloc[0]), float(cos_m.iloc[0])

        # Flatten spatial arrays
        flat = {
            "time_idx"          : t_idx,
            "year"              : yr,
            "month"             : m,
            "month_sin"         : sin_m,
            "month_cos"         : cos_m,
            "months_since_start": t0,
            "lat"               : lat_grid.ravel(),
            "lon"               : lon_grid.ravel(),
            "lat_norm"          : ((lat_grid - lat_mean) / lat_std).ravel(),
            "lon_norm"          : ((lon_grid - lon_mean) / lon_std).ravel(),
            "climatology_mm"    : clim_arr[m - 1].ravel(),    # m is 1-indexed
            "anomaly_lag1"      : anom_lag1[t_idx].ravel(),
            "anomaly_lag2"      : anom_lag2[t_idx].ravel(),
            "anomaly_lag3"      : anom_lag3[t_idx].ravel(),
            "precip_lag1"       : precip_lag1[t_idx].ravel(),
            "precip_lag2"       : precip_lag2[t_idx].ravel(),
            "rolling_mean_3m"   : roll_mean3[t_idx].ravel(),
            "rolling_std_3m"    : roll_std3[t_idx].ravel(),
            "rolling_mean_6m"   : roll_mean6[t_idx].ravel(),
            "anomaly_mm"        : anom_arr[t_idx].ravel(),     # TARGET
        }

        n_cells = n_lat * n_lon
        # Broadcast scalar features to match n_cells
        scalars = ["time_idx", "year", "month", "month_sin", "month_cos",
                   "months_since_start"]
        df_t = pd.DataFrame({k: np.full(n_cells, flat[k], dtype=np.float32)
                             if k in scalars else flat[k]
                             for k in flat})
        rows.append(df_t)

    df_all = pd.concat(rows, ignore_index=True)
    if verbose:
        print(f"   Total rows (before NaN drop): {len(df_all):,}")

    # ---- 5. Drop rows with NaN lag features ---------------------------------
    if verbose:
        print("\n[5/6] Dropping rows with NaN lag features...")

    feature_cols = [
        "year", "month", "month_sin", "month_cos", "months_since_start",
        "lat", "lon", "lat_norm", "lon_norm",
        "climatology_mm",
        "anomaly_lag1", "anomaly_lag2", "anomaly_lag3",
        "precip_lag1", "precip_lag2",
        "rolling_mean_3m", "rolling_std_3m",
        "rolling_mean_6m",
    ]
    target_col = "anomaly_mm"

    # NaN lag features exist for the first max_lag months (0..5 for roll_mean6)
    # We use the most conservative: drop any row where ANY feature is NaN
    n_before = len(df_all)
    df_all.dropna(subset=feature_cols + [target_col], inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    n_after = len(df_all)
    months_dropped = (n_before - n_after) // (n_lat * n_lon)
    if verbose:
        print(f"   Dropped {n_before - n_after:,} rows "
              f"(first {months_dropped} months have incomplete lag windows)")
        print(f"   Remaining rows: {n_after:,}")

    # ---- 6. Chronological train / val / test split --------------------------
    if verbose:
        print("\n[6/6] Chronological train / val / test split...")

    # Monsoon-aware chronological split
    # ---------------------------------------------------------------
    # The split ensures each set contains full calendar years to
    # capture the full Indian monsoon cycle and non-monsoon periods.
    #
    # Train: t=6..131  (Jul 2010 - Dec 2020) | 126 months
    #   Provides: 10 full monsoon seasons for the model to learn
    #   long-term interannual variability.
    #
    # Val:   t=132..155 (Jan 2021 - Dec 2022) | 24 months
    #   Contains: 2 full years (2 monsoons) for robust early stopping
    #   and hyperparameter tuning.
    #
    # Test:  t=156..179 (Jan 2023 - Dec 2024) | 24 months
    #   Contains: 2 full years (2 monsoons) held-out, never seen.
    #   Used for: FINAL evaluation only (run once).

    # Fixed explicit time index assignments (after NaN drop, t=6..179 available)
    remaining_time_idxs = sorted(int(x) for x in df_all["time_idx"].unique())
    train_idxs = [t for t in remaining_time_idxs if t <= 131]
    val_idxs   = [t for t in remaining_time_idxs if 132 <= t <= 155]
    test_idxs  = [t for t in remaining_time_idxs if 156 <= t <= 179]



    df_train = df_all[df_all["time_idx"].isin(train_idxs)].copy()
    df_val   = df_all[df_all["time_idx"].isin(val_idxs)].copy()
    df_test  = df_all[df_all["time_idx"].isin(test_idxs)].copy()

    # Compute date ranges from time_idxs
    def idx_to_dates(idxs):
        return (str(times[int(idxs[0])])[:10], str(times[int(idxs[-1])])[:10])

    train_range = idx_to_dates(train_idxs)
    val_range   = idx_to_dates(val_idxs)
    test_range  = idx_to_dates(test_idxs)

    if verbose:
        print(f"   Train: {len(train_idxs)} months ({train_range[0]} -> {train_range[1]})"
              f"  -> {len(df_train):,} rows")
        print(f"   Val  : {len(val_idxs)} months ({val_range[0]} -> {val_range[1]})"
              f"  -> {len(df_val):,} rows")
        print(f"   Test : {len(test_idxs)} months ({test_range[0]} -> {test_range[1]})"
              f"  -> {len(df_test):,} rows")

    # ---- StandardScaler fit on TRAIN only -----------------------------------
    # Compute mean and std from train rows only
    scaler_params = {}
    for col in feature_cols:
        mu  = df_train[col].mean()
        std = df_train[col].std()
        if std == 0:
            std = 1.0   # avoid division by zero for constant features
        scaler_params[col] = {"mean": mu, "std": std}

    # Apply scaling (creates new scaled columns, keep originals too)
    for col in feature_cols:
        mu  = scaler_params[col]["mean"]
        std = scaler_params[col]["std"]
        col_s = col + "_scaled"
        df_train[col_s] = (df_train[col] - mu) / std
        df_val[col_s]   = (df_val[col]   - mu) / std
        df_test[col_s]  = (df_test[col]  - mu) / std

    scaled_feature_cols = [c + "_scaled" for c in feature_cols]

    # ---- 7. Save outputs ----------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(OUT_DIR / "features_train.parquet", index=False)
    df_val.to_parquet(  OUT_DIR / "features_val.parquet",   index=False)
    df_test.to_parquet( OUT_DIR / "features_test.parquet",  index=False)

    # Feature names
    with open(OUT_DIR / "feature_names.txt", "w") as f:
        f.write("RAW FEATURE COLUMNS:\n")
        for c in feature_cols:
            f.write(f"  {c}\n")
        f.write("\nSCALED FEATURE COLUMNS (use these for model input):\n")
        for c in scaled_feature_cols:
            f.write(f"  {c}\n")
        f.write(f"\nTARGET COLUMN: {target_col}\n")

    # Split summary
    with open(OUT_DIR / "split_summary.txt", "w") as f:
        f.write("SPLIT SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total months (after NaN drop) : {len(remaining_time_idxs)}\n")
        f.write(f"First {months_dropped} months dropped (incomplete lag windows)\n\n")
        f.write(f"TRAIN : {len(train_idxs)} months | "
                f"{train_range[0]} to {train_range[1]} | {len(df_train):,} rows\n")
        f.write(f"VAL   : {len(val_idxs)} months | "
                f"{val_range[0]} to {val_range[1]} | {len(df_val):,} rows\n")
        f.write(f"TEST  : {len(test_idxs)} months | "
                f"{test_range[0]} to {test_range[1]} | {len(df_test):,} rows\n\n")
        f.write(f"FEATURES ({len(feature_cols)} raw, {len(scaled_feature_cols)} scaled):\n")
        for c in feature_cols:
            f.write(f"  {c}\n")
        f.write(f"\nTARGET: {target_col}\n")
        f.write(f"\nSCALER: fit on TRAIN only (mean/std per feature)\n")
        f.write("\nSCALER PARAMS (train):\n")
        for col, params in scaler_params.items():
            f.write(f"  {col:25s}: mean={params['mean']:.4f}, std={params['std']:.4f}\n")

    if verbose:
        print(f"\n   Saved to: {OUT_DIR}/")
        print(f"     features_train.parquet  ({len(df_train):,} rows)")
        print(f"     features_val.parquet    ({len(df_val):,} rows)")
        print(f"     features_test.parquet   ({len(df_test):,} rows)")
        print(f"     feature_names.txt")
        print(f"     split_summary.txt")

    return {
        "df_train"           : df_train,
        "df_val"             : df_val,
        "df_test"            : df_test,
        "feature_cols"       : feature_cols,
        "scaled_feature_cols": scaled_feature_cols,
        "target_col"         : target_col,
        "scaler_params"      : scaler_params,
        "train_range"        : train_range,
        "val_range"          : val_range,
        "test_range"         : test_range,
        "n_months_dropped"   : months_dropped,
    }


# ---- diagnostics ------------------------------------------------------------

def print_feature_summary(result: dict) -> None:
    """Print a human-readable summary of the feature matrix."""
    df_train = result["df_train"]
    df_val   = result["df_val"]
    df_test  = result["df_test"]
    feat     = result["feature_cols"]
    target   = result["target_col"]

    print("\n" + "=" * 60)
    print("  FEATURE MATRIX SUMMARY")
    print("=" * 60)

    print(f"\nNumber of features    : {len(feat)}")
    print(f"Target variable       : {target}  (continuous mm anomaly)")
    print(f"\nFeature names:")
    for c in feat:
        print(f"  {c}")

    print(f"\nSplit row counts:")
    print(f"  Train : {len(df_train):,} rows  ({result['train_range'][0]} -> {result['train_range'][1]})")
    print(f"  Val   : {len(df_val):,} rows  ({result['val_range'][0]} -> {result['val_range'][1]})")
    print(f"  Test  : {len(df_test):,} rows  ({result['test_range'][0]} -> {result['test_range'][1]})")
    print(f"  Total : {len(df_train) + len(df_val) + len(df_test):,} rows")

    print(f"\nMissing values:")
    for col in feat + [target]:
        n_nan_train = df_train[col].isna().sum()
        n_nan_val   = df_val[col].isna().sum()
        n_nan_test  = df_test[col].isna().sum()
        if n_nan_train + n_nan_val + n_nan_test > 0:
            print(f"  WARNING: {col}  train={n_nan_train} val={n_nan_val} test={n_nan_test}")
    total_nan = sum(
        (df_train[c].isna().sum() + df_val[c].isna().sum() + df_test[c].isna().sum())
        for c in feat + [target]
    )
    if total_nan == 0:
        print("  None found — clean dataset.")

    print(f"\nTarget statistics (train):")
    target_train = df_train[target]
    print(f"  min    : {target_train.min():.2f} mm")
    print(f"  max    : {target_train.max():.2f} mm")
    print(f"  mean   : {target_train.mean():.4f} mm")
    print(f"  std    : {target_train.std():.2f} mm")
    print(f"  median : {target_train.median():.2f} mm")


# ---- entry point ------------------------------------------------------------

if __name__ == "__main__":
    result = build_feature_matrix(verbose=True)
    print_feature_summary(result)
    print("\n[features.py] Pipeline COMPLETE.")
