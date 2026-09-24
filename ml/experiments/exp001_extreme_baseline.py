"""
ml/experiments/exp001_extreme_baseline.py
------------------------------------------
EXP-001: Statistical Extreme-Event Baseline

PURPOSE
=======
Detect extreme precipitation anomaly events using purely statistical thresholds.
No machine learning. No training loop. No forecasting.

This experiment retrospectively labels each (time, lat, lon) grid cell as
"extreme" or "normal" based on whether its anomaly magnitude exceeds 1.5
standard deviations of the local (per-cell) variability.

DEFINITION
==========
  extreme := |anomaly(t, lat, lon)| > 1.5 * std_dev(lat, lon)

  where std_dev(lat, lon) is computed across all 28 available monthly timesteps.

  Positive extreme: anomaly > +1.5 * std_dev  (excess / flood risk)
  Negative extreme: anomaly < -1.5 * std_dev  (deficit / drought risk)

INPUTS
======
  precipitation_anomaly.nc  (28 x 125 x 121, existing derived file)

OUTPUTS
=======
  ml/experiments/exp001_extreme_masks.nc   -- binary extreme mask NetCDF
  ml/experiments/exp001_stats.json          -- per-timestep and spatial statistics
  ml/experiments/exp001_extreme_map.png     -- visualization of extreme events

SCIENTIFIC NOTES
================
  - This is NOT a forecast. All thresholds are derived from the same 28-month
    record being analyzed (retrospective/diagnostic, not predictive).
  - std_dev computed from 28 months is NOT a climatological normal (WMO standard
    is 30 years). Interpret thresholds with caution.
  - The 1.5-sigma threshold corresponds to ~86.6% confidence under Gaussian
    assumption. Precipitation is not Gaussian, so the actual extreme fraction
    will differ from the theoretical 13.4%.
  - No ground truth validation is performed. No model accuracy is claimed.
"""

import pathlib
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# ---- paths ------------------------------------------------------------------
ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent
ANOM_PATH = ROOT / "precipitation_anomaly.nc"
OUT_DIR   = ROOT / "ml" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_PATH = OUT_DIR / "exp001_extreme_masks.nc"
STATS_PATH = OUT_DIR / "exp001_stats.json"
VIZ_PATH  = OUT_DIR / "exp001_extreme_map.png"

THRESHOLD_SIGMA = 1.5


# ---- helpers ----------------------------------------------------------------

def season_of_month(month: int) -> str:
    """Return Indian meteorological season for a given calendar month."""
    if month in (12, 1, 2):
        return "DJF (Winter)"
    elif month in (3, 4, 5):
        return "MAM (Pre-monsoon)"
    elif month in (6, 7, 8, 9):
        return "JJAS (Monsoon)"
    else:  # 10, 11
        return "ON (Post-monsoon)"


# ---- main pipeline ----------------------------------------------------------

def run_exp001(verbose: bool = True) -> dict:
    print("=" * 65)
    print("  EXP-001: STATISTICAL EXTREME-EVENT BASELINE")
    print("=" * 65)
    print(f"  Threshold : |anomaly| > {THRESHOLD_SIGMA} * local_std")
    print(f"  Input     : {ANOM_PATH}")
    print(f"  Output dir: {OUT_DIR}")
    print()

    # ---- 1. Load anomaly dataset --------------------------------------------
    if verbose:
        print("[1/7] Loading precipitation_anomaly.nc ...")
    ds = xr.open_dataset(ANOM_PATH)

    # The variable is 'tp' in the anomaly file
    anom_var = list(ds.data_vars)[0]
    if verbose:
        print(f"   Variable: '{anom_var}'")
        print(f"   Dims    : {dict(ds.dims)}")
        print(f"   Coords  : {list(ds.coords)}")

    anom = ds[anom_var]   # DataArray (valid_time, latitude, longitude)

    times = pd.DatetimeIndex(anom.valid_time.values)
    lats  = anom.latitude.values
    lons  = anom.longitude.values
    n_t   = len(times)
    n_lat = len(lats)
    n_lon = len(lons)
    n_cells = n_lat * n_lon

    if verbose:
        print(f"   Time range: {times[0].date()} to {times[-1].date()} ({n_t} months)")
        print(f"   Grid      : {n_lat} lat × {n_lon} lon = {n_cells:,} cells")

    anom_arr = anom.values  # (28, 125, 121) numpy float32/64

    # ---- 2. Compute per-cell standard deviation across all timesteps --------
    if verbose:
        print("\n[2/7] Computing per-cell standard deviation ...")

    # ddof=1: unbiased estimate (though 28 samples is still small)
    cell_std  = np.std(anom_arr, axis=0, ddof=1)   # (125, 121)
    cell_mean = np.mean(anom_arr, axis=0)            # (125, 121)

    # Threshold per cell
    threshold = THRESHOLD_SIGMA * cell_std           # (125, 121)

    if verbose:
        print(f"   cell_std  : min={cell_std.min():.2f} mm, "
              f"max={cell_std.max():.2f} mm, "
              f"mean={cell_std.mean():.2f} mm")
        print(f"   threshold : min={threshold.min():.2f} mm, "
              f"max={threshold.max():.2f} mm, "
              f"mean={threshold.mean():.2f} mm")

    # ---- 3. Create binary extreme masks -------------------------------------
    if verbose:
        print("\n[3/7] Creating binary extreme masks ...")

    # pos_extreme: anomaly > +1.5*std  (wet extreme)
    # neg_extreme: anomaly < -1.5*std  (dry extreme)
    # extreme_mask: 1 where |anomaly| > threshold, 0 otherwise
    pos_mask  = (anom_arr >  threshold[np.newaxis, :, :]).astype(np.int8)  # (28,125,121)
    neg_mask  = (anom_arr < -threshold[np.newaxis, :, :]).astype(np.int8)
    ext_mask  = (pos_mask | neg_mask).astype(np.int8)   # 0 or 1

    # Signed mask: +1 (pos extreme), -1 (neg extreme), 0 (normal)
    signed_mask = pos_mask.astype(np.int8) - neg_mask.astype(np.int8)

    total_extreme_cells = int(ext_mask.sum())
    total_pos_cells     = int(pos_mask.sum())
    total_neg_cells     = int(neg_mask.sum())

    if verbose:
        print(f"   Total extreme cell-months : {total_extreme_cells:,} "
              f"({100*total_extreme_cells/(n_t*n_cells):.2f}%)")
        print(f"   Positive extreme (wet)    : {total_pos_cells:,}")
        print(f"   Negative extreme (dry)    : {total_neg_cells:,}")

    # ---- 4. Build per-timestep statistics -----------------------------------
    if verbose:
        print("\n[4/7] Computing per-timestep statistics ...")

    timestep_stats = []
    for t_idx in range(n_t):
        ts = times[t_idx]
        n_ext     = int(ext_mask[t_idx].sum())
        n_pos     = int(pos_mask[t_idx].sum())
        n_neg     = int(neg_mask[t_idx].sum())
        pct_ext   = round(100 * n_ext / n_cells, 4)
        pct_pos   = round(100 * n_pos / n_cells, 4)
        pct_neg   = round(100 * n_neg / n_cells, 4)
        anom_t    = anom_arr[t_idx]
        ext_cells = anom_t[ext_mask[t_idx] == 1]

        timestep_stats.append({
            "time_idx"             : t_idx,
            "date"                 : str(ts.date()),
            "month"                : int(ts.month),
            "year"                 : int(ts.year),
            "season"               : season_of_month(int(ts.month)),
            "n_extreme_cells"      : n_ext,
            "n_positive_extreme"   : n_pos,
            "n_negative_extreme"   : n_neg,
            "pct_extreme"          : pct_ext,
            "pct_positive_extreme" : pct_pos,
            "pct_negative_extreme" : pct_neg,
            "anom_mean_all_mm"     : round(float(anom_t.mean()), 4),
            "anom_std_all_mm"      : round(float(anom_t.std()),  4),
            "anom_mean_extreme_mm" : round(float(ext_cells.mean()), 4) if len(ext_cells) > 0 else None,
            "anom_max_mm"          : round(float(anom_t.max()), 4),
            "anom_min_mm"          : round(float(anom_t.min()), 4),
        })

    # ---- 5. Seasonal and spatial summary ------------------------------------
    if verbose:
        print("\n[5/7] Computing seasonal summary ...")

    df_ts = pd.DataFrame(timestep_stats)

    # Season grouping
    seasonal_summary = {}
    for season, grp in df_ts.groupby("season"):
        seasonal_summary[season] = {
            "n_timesteps"            : int(len(grp)),
            "mean_pct_extreme"       : round(float(grp["pct_extreme"].mean()), 4),
            "mean_pct_positive"      : round(float(grp["pct_positive_extreme"].mean()), 4),
            "mean_pct_negative"      : round(float(grp["pct_negative_extreme"].mean()), 4),
            "max_pct_extreme"        : round(float(grp["pct_extreme"].max()), 4),
            "total_extreme_cells"    : int(grp["n_extreme_cells"].sum()),
        }

    # Monthly mean (across years)
    monthly_summary = {}
    for month, grp in df_ts.groupby("month"):
        monthly_summary[int(month)] = {
            "n_occurrences"      : int(len(grp)),
            "mean_pct_extreme"   : round(float(grp["pct_extreme"].mean()), 4),
            "mean_pct_positive"  : round(float(grp["pct_positive_extreme"].mean()), 4),
            "mean_pct_negative"  : round(float(grp["pct_negative_extreme"].mean()), 4),
        }

    # Spatial summary: frequency of extremes per cell across all timesteps
    freq_extreme  = ext_mask.sum(axis=0)  / n_t  # (125,121) 0..1
    freq_positive = pos_mask.sum(axis=0)  / n_t
    freq_negative = neg_mask.sum(axis=0)  / n_t

    hot_cells_lat_idx, hot_cells_lon_idx = np.where(freq_extreme >= 0.3)
    top_cells = [
        {
            "lat": round(float(lats[i]), 2),
            "lon": round(float(lons[j]), 2),
            "freq_extreme": round(float(freq_extreme[i, j]), 4),
        }
        for i, j in zip(hot_cells_lat_idx, hot_cells_lon_idx)
    ]
    top_cells.sort(key=lambda x: -x["freq_extreme"])

    # ---- 6. Save stats JSON -------------------------------------------------
    if verbose:
        print("\n[6/7] Saving statistics JSON ...")

    stats = {
        "experiment"        : "EXP-001",
        "description"       : "Statistical extreme-event baseline (no ML, no forecasting)",
        "generated_at"      : datetime.utcnow().isoformat() + "Z",
        "inputs"            : {"anomaly_file": str(ANOM_PATH)},
        "threshold"         : {
            "sigma_multiplier"  : THRESHOLD_SIGMA,
            "definition"        : f"|anomaly| > {THRESHOLD_SIGMA} * per-cell std-dev",
            "std_computed_over" : f"All {n_t} available monthly timesteps (ddof=1)",
            "caution"           : (
                "Threshold derived from same 28-month record being analyzed. "
                "This is retrospective (diagnostic), NOT a forecast. "
                "28 months is not a climatological normal (WMO standard = 30 years)."
            ),
        },
        "grid"              : {
            "n_timesteps"   : n_t,
            "n_lat"         : n_lat,
            "n_lon"         : n_lon,
            "n_cells"       : n_cells,
            "time_range"    : [str(times[0].date()), str(times[-1].date())],
            "lat_range"     : [round(float(lats.min()), 2), round(float(lats.max()), 2)],
            "lon_range"     : [round(float(lons.min()), 2), round(float(lons.max()), 2)],
        },
        "global_summary"    : {
            "total_extreme_cell_months"   : total_extreme_cells,
            "total_positive_extreme"      : total_pos_cells,
            "total_negative_extreme"      : total_neg_cells,
            "overall_pct_extreme"         : round(100 * total_extreme_cells / (n_t * n_cells), 4),
            "overall_pct_positive"        : round(100 * total_pos_cells     / (n_t * n_cells), 4),
            "overall_pct_negative"        : round(100 * total_neg_cells     / (n_t * n_cells), 4),
            "cell_std_mean_mm"            : round(float(cell_std.mean()), 4),
            "cell_std_max_mm"             : round(float(cell_std.max()),  4),
            "cell_std_min_mm"             : round(float(cell_std.min()),  4),
        },
        "timestep_stats"    : timestep_stats,
        "seasonal_summary"  : seasonal_summary,
        "monthly_summary"   : monthly_summary,
        "spatial_hotspots"  : {
            "definition"    : "Grid cells where extreme-event frequency >= 30% of timesteps",
            "n_hotspot_cells": len(top_cells),
            "top_cells"     : top_cells[:20],
        },
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   Saved: {STATS_PATH}")

    # ---- 7. Save NetCDF mask ------------------------------------------------
    if verbose:
        print("\n[7a/7] Saving extreme mask NetCDF ...")

    ds_out = xr.Dataset(
        {
            "extreme_mask": xr.DataArray(
                ext_mask,
                dims=["valid_time", "latitude", "longitude"],
                coords={
                    "valid_time": anom.valid_time.values,
                    "latitude":   lats,
                    "longitude":  lons,
                },
                attrs={
                    "long_name"  : "Binary extreme precipitation anomaly mask",
                    "description": (
                        f"1 = |anomaly| > {THRESHOLD_SIGMA} * local_std,  0 = normal"
                    ),
                    "units"      : "0 or 1 (dimensionless)",
                    "threshold_sigma": THRESHOLD_SIGMA,
                },
            ),
            "signed_extreme_mask": xr.DataArray(
                signed_mask,
                dims=["valid_time", "latitude", "longitude"],
                coords={
                    "valid_time": anom.valid_time.values,
                    "latitude":   lats,
                    "longitude":  lons,
                },
                attrs={
                    "long_name"  : "Signed extreme mask: +1=wet extreme, -1=dry extreme, 0=normal",
                    "units"      : "-1, 0, or 1 (dimensionless)",
                },
            ),
            "local_std": xr.DataArray(
                cell_std,
                dims=["latitude", "longitude"],
                coords={
                    "latitude":  lats,
                    "longitude": lons,
                },
                attrs={
                    "long_name": "Per-cell standard deviation of monthly anomaly (all 28 months, ddof=1)",
                    "units"    : "mm",
                },
            ),
            "local_threshold": xr.DataArray(
                threshold,
                dims=["latitude", "longitude"],
                coords={
                    "latitude":  lats,
                    "longitude": lons,
                },
                attrs={
                    "long_name": f"Per-cell threshold = {THRESHOLD_SIGMA} * local_std",
                    "units"    : "mm",
                },
            ),
        },
        attrs={
            "experiment"    : "EXP-001 Statistical Extreme-Event Baseline",
            "project"       : "VarshaVani — SIH AI-Driven Extreme Weather Anomaly Tracking",
            "source_file"   : "precipitation_anomaly.nc",
            "threshold_def" : f"|anomaly| > {THRESHOLD_SIGMA} * per-cell std-dev",
            "caution"       : "Retrospective diagnostic. NOT a forecast.",
            "created_at"    : datetime.utcnow().isoformat() + "Z",
        },
    )

    ds_out.to_netcdf(MASK_PATH)
    print(f"   Saved: {MASK_PATH}")

    # ---- 8. Verify round-trip -----------------------------------------------
    if verbose:
        print("\n[7b/7] Verifying saved NetCDF (round-trip check) ...")
    ds_check = xr.open_dataset(MASK_PATH)

    assert dict(ds_check.dims) == {
        "valid_time": n_t, "latitude": n_lat, "longitude": n_lon
    }, "Dimension mismatch on reload!"
    assert set(ds_check.data_vars) == {
        "extreme_mask", "signed_extreme_mask", "local_std", "local_threshold"
    }, "Variable mismatch on reload!"
    np.testing.assert_array_equal(
        ds_check["extreme_mask"].values, ext_mask,
        err_msg="extreme_mask roundtrip failed"
    )
    print(f"   Round-trip verification PASSED.")
    print(f"   Reloaded dims: {dict(ds_check.dims)}")
    print(f"   Reloaded vars: {list(ds_check.data_vars)}")
    ds_check.close()
    ds.close()

    # ---- 9. Visualization ---------------------------------------------------
    if verbose:
        print("\n[7c/7] Creating visualization ...")

    _make_visualization(
        anom_arr, ext_mask, pos_mask, neg_mask, signed_mask,
        freq_extreme, times, lats, lons, df_ts, cell_std
    )

    # ---- 10. Final summary --------------------------------------------------
    print()
    print("=" * 65)
    print("  EXP-001 COMPLETE")
    print("=" * 65)
    print(f"  Threshold           : |anomaly| > {THRESHOLD_SIGMA}*sigma per cell")
    print(f"  Total extreme cells : {total_extreme_cells:,} "
          f"({100*total_extreme_cells/(n_t*n_cells):.2f}% of all cell-months)")
    print(f"  Wet extremes        : {total_pos_cells:,}")
    print(f"  Dry extremes        : {total_neg_cells:,}")
    print(f"  Hotspot cells (>=30%): {len(top_cells):,}")
    print(f"  Outputs:")
    print(f"    {MASK_PATH}")
    print(f"    {STATS_PATH}")
    print(f"    {VIZ_PATH}")
    print()
    print("  CAUTION: This is a retrospective analysis on 28 months of data.")
    print("  It is NOT a forecast. No model accuracy is claimed.")

    return stats


# ---- visualization ----------------------------------------------------------

def _make_visualization(
    anom_arr, ext_mask, pos_mask, neg_mask, signed_mask,
    freq_extreme, times, lats, lons, df_ts, cell_std
):
    """
    4-panel figure:
      (a) Top-left:  Monsoon month frequency of extreme events (spatial map)
      (b) Top-right: Per-cell local standard deviation (spatial map)
      (c) Bottom-left: Time series of % extreme cells per month
      (d) Bottom-right: Last monsoon month (Sep 2022) signed extreme mask
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    cmap_freq  = plt.cm.YlOrRd
    cmap_std   = plt.cm.plasma

    # ---- (a) Frequency of extreme events across all 28 months ---------------
    ax = axes[0, 0]
    im = ax.pcolormesh(lon_grid, lat_grid, freq_extreme,
                       cmap=cmap_freq, vmin=0, vmax=0.5, shading="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.label.set_color("white")
    cb.ax.tick_params(colors="white")
    cb.set_label("Fraction of months with extreme anomaly", color="white", fontsize=9)
    ax.set_title("Extreme Event Frequency\n(|anomaly| > 1.5σ, all 28 months)",
                 color="white", fontsize=11)
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=9)
    ax.set_ylabel("Latitude (°N)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ---- (b) Per-cell standard deviation map --------------------------------
    ax = axes[0, 1]
    im2 = ax.pcolormesh(lon_grid, lat_grid, cell_std,
                        cmap=cmap_std, vmin=0, shading="auto")
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.ax.yaxis.label.set_color("white")
    cb2.ax.tick_params(colors="white")
    cb2.set_label("Local std-dev (mm)", color="white", fontsize=9)
    ax.set_title("Per-Cell Anomaly Std-Dev\n(basis for thresholds, 28 months)",
                 color="white", fontsize=11)
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=9)
    ax.set_ylabel("Latitude (°N)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ---- (c) Time series: % extreme cells per month -------------------------
    ax = axes[1, 0]
    date_labels = [str(times[i])[:7] for i in range(len(times))]
    x = range(len(times))

    pct_pos = df_ts["pct_positive_extreme"].values
    pct_neg = df_ts["pct_negative_extreme"].values

    ax.bar(x, pct_pos, color="#f97316", alpha=0.85, label="Wet extreme (pos)")
    ax.bar(x, -pct_neg, color="#3b82f6", alpha=0.85, label="Dry extreme (neg)")
    ax.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.5)

    # Shade monsoon months (Jun-Sep) in subtle green
    for i, ts in enumerate(times):
        if ts.month in (6, 7, 8, 9):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="#22c55e", zorder=0)

    step = max(1, len(times) // 14)
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels([date_labels[i] for i in range(0, len(times), step)],
                       rotation=45, ha="right", color="white", fontsize=7)
    ax.set_ylabel("% of grid cells", color="white", fontsize=9)
    ax.set_title("% Extreme Cells per Month\n(green shading = JJAS monsoon months)",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.legend(fontsize=8, facecolor="#1c2128", labelcolor="white", framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ---- (d) Signed mask for the last month (Sep 2022) ----------------------
    ax = axes[1, 1]
    last_signed = signed_mask[-1]  # (125, 121)
    last_date   = str(times[-1])[:7]

    cmap_signed = mcolors.LinearSegmentedColormap.from_list(
        "signed", ["#3b82f6", "#161b22", "#f97316"], N=3
    )
    im3 = ax.pcolormesh(lon_grid, lat_grid, last_signed,
                        cmap=cmap_signed, vmin=-1, vmax=1, shading="auto")
    cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04,
                       ticks=[-1, 0, 1])
    cb3.ax.set_yticklabels(["Dry extreme", "Normal", "Wet extreme"],
                           color="white", fontsize=8)
    cb3.ax.tick_params(colors="white")
    ax.set_title(f"Signed Extreme Mask — {last_date}\n"
                 f"(+1=wet extreme, -1=dry extreme, 0=normal)",
                 color="white", fontsize=11)
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=9)
    ax.set_ylabel("Latitude (°N)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    fig.suptitle(
        "EXP-001: Statistical Extreme Precipitation Anomaly Baseline\n"
        "India Domain | ERA5 Monthly Anomaly | Jun 2020 – Sep 2022\n"
        "Threshold: |anomaly| > 1.5σ (per-cell, retrospective diagnostic — NOT a forecast)",
        color="white", fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(VIZ_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Saved: {VIZ_PATH}")


# ---- entry point ------------------------------------------------------------

if __name__ == "__main__":
    stats = run_exp001(verbose=True)
