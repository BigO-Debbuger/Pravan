"""
ml/experiments/exp002_persistence_baseline.py
----------------------------------------------
EXP-002: Persistence Baseline

PURPOSE
=======
Establish the simplest possible dynamic forecast floor:
predict next month's anomaly by repeating the previous month's observed anomaly.

This is the standard meteorological persistence baseline. Any ML model that
cannot beat persistence has no practical forecasting value.

PREDICTION DEFINITION (no leakage)
====================================
  prediction(t, lat, lon) = anomaly_observed(t-1, lat, lon)
  target(t, lat, lon)     = anomaly_observed(t, lat, lon)
  horizon                 = 1 month ahead (t-1 -> t)

  The prediction for month t is formed ONLY from the anomaly at t-1.
  No information from month t or later is used. There is zero look-ahead.

SPLITS EVALUATED
================
  Val  : Jun 2022, Jul 2022  (2 monsoon months, early monsoon)
  Test : Aug 2022, Sep 2022  (2 monsoon months, held-out)

  These are the same splits used in EXP-003 (XGBoost) to enable direct
  comparison. The persistence baseline provides the skill floor against
  which all ML models should be compared.

  Note: persistence requires anomaly(t-1) as input, so month t=Jun 2022
  uses anomaly(t-1)=May 2022. May 2022 is the last training month, so
  no training data is contaminated.

METRICS
=======
  - MAE  : Mean Absolute Error
  - RMSE : Root Mean Square Error
  - R2   : R-squared (coefficient of determination)
           NOTE: R2 can be negative when predictions are worse than the mean.
  - ACC  : Anomaly Correlation Coefficient
           ACC = corr(pred - clim, obs - clim) across spatial grid per timestep
           WMO threshold for skillful forecasts: ACC > 0.6
  - Bias : mean(pred) - mean(obs)
  - POD, FAR, CSI : Extreme-event detection skill using EXP-001 threshold
           These require the extreme mask from EXP-001 as ground truth.

EVENT-BASED METRICS (for extreme anomaly detection)
===================================================
  Using EXP-001 threshold (|anomaly| > 1.5 * per-cell std):
  - If persistence predicts an extreme at t-1 and an extreme actually
    occurs at t, this counts as a hit (TP).
  - Persistence of the SIGNED mask (+1/-+0/-1) tests whether wet/dry
    extremes persist month-to-month.

INPUTS
======
  precipitation_anomaly.nc          (28 months, 125 x 121)
  ml/processed/features_val.parquet  (for split time indices)
  ml/processed/features_test.parquet
  ml/experiments/exp001_extreme_masks.nc  (for event-based metrics)
  ml/models/scaler_params.json       (for climatology reconstruction, if needed)

OUTPUTS
=======
  ml/experiments/exp002_persistence_metrics.json   -- all metrics
  ml/experiments/exp002_persistence_predictions.nc -- predictions + actuals
  ml/experiments/exp002_persistence_plot.png       -- visualization

SCIENTIFIC NOTES
================
  - This is NOT a trained model. It has zero free parameters.
  - Persistence is typically the weakest useful baseline for weather prediction.
  - Precipitation anomalies are generally weakly persistent month-to-month,
    especially across monsoon season boundaries.
  - Do NOT interpret these results as model accuracy or forecast skill.
"""

import pathlib
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ---- paths ------------------------------------------------------------------
ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent
ANOM_PATH = ROOT / "precipitation_anomaly.nc"
MASK_PATH = ROOT / "ml" / "experiments" / "exp001_extreme_masks.nc"
PROC_DIR  = ROOT / "ml" / "processed"
OUT_DIR   = ROOT / "ml" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH     = OUT_DIR / "exp002_persistence_metrics.json"
PRED_NC_PATH     = OUT_DIR / "exp002_persistence_predictions.nc"
PLOT_PATH        = OUT_DIR / "exp002_persistence_plot.png"

# Same time index assignments as features.py (after 6-month NaN drop)
# Train: t=6..23  (Dec 2020 - May 2022)
# Val:   t=24,25  (Jun 2022, Jul 2022)
# Test:  t=26,27  (Aug 2022, Sep 2022)
VAL_TIDXS  = [24, 25]
TEST_TIDXS = [26, 27]

EXTREME_SIGMA = 1.5


# ---- helpers ----------------------------------------------------------------

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2; returns None if SS_tot == 0 (undefined for constant target)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return None
    return float(1.0 - ss_res / ss_tot)


def anomaly_correlation_coefficient(pred: np.ndarray, obs: np.ndarray,
                                    clim: np.ndarray) -> float:
    """
    Spatial ACC for a single timestep map.
    ACC = Pearson corr between (pred - clim) and (obs - clim) across all cells.
    Returns NaN if either anomaly field has zero variance.
    """
    pred_a = pred - clim
    obs_a  = obs  - clim
    if pred_a.std() == 0 or obs_a.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred_a.ravel(), obs_a.ravel())[0, 1])


def event_metrics(pred_extreme: np.ndarray, obs_extreme: np.ndarray):
    """
    Compute POD, FAR, CSI for binary extreme masks.
    pred_extreme, obs_extreme: boolean/int arrays of same shape (0 or 1).
    """
    TP = int(((pred_extreme == 1) & (obs_extreme == 1)).sum())
    FP = int(((pred_extreme == 1) & (obs_extreme == 0)).sum())
    FN = int(((pred_extreme == 0) & (obs_extreme == 1)).sum())
    TN = int(((pred_extreme == 0) & (obs_extreme == 0)).sum())

    pod = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    far = FP / (FP + TP) if (FP + TP) > 0 else float("nan")
    csi = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else float("nan")

    # Equitable Threat Score
    n_total = TP + FP + FN + TN
    hits_random = (TP + FP) * (TP + FN) / n_total if n_total > 0 else 0
    ets_denom = TP + FP + FN - hits_random
    ets = (TP - hits_random) / ets_denom if ets_denom != 0 else float("nan")

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "POD": round(pod, 4) if not np.isnan(pod) else None,
            "FAR": round(far, 4) if not np.isnan(far) else None,
            "CSI": round(csi, 4) if not np.isnan(csi) else None,
            "ETS": round(ets, 4) if not np.isnan(ets) else None}


# ---- main pipeline ----------------------------------------------------------

def run_exp002(verbose: bool = True) -> dict:
    print("=" * 65)
    print("  EXP-002: PERSISTENCE BASELINE")
    print("=" * 65)
    print("  Strategy : prediction(t) = anomaly(t-1)  [1-month persistence]")
    print(f"  Input    : {ANOM_PATH.name}")
    print()

    # ---- 1. Load anomaly dataset --------------------------------------------
    if verbose:
        print("[1/8] Loading precipitation_anomaly.nc ...")
    ds_anom = xr.open_dataset(ANOM_PATH)
    anom_var  = list(ds_anom.data_vars)[0]   # 'tp'
    anom_da   = ds_anom[anom_var]             # (28, 125, 121)
    anom_arr  = anom_da.values               # numpy (28, 125, 121)

    times  = pd.DatetimeIndex(anom_da.valid_time.values)
    lats   = anom_da.latitude.values
    lons   = anom_da.longitude.values
    n_t    = len(times)
    n_lat  = len(lats)
    n_lon  = len(lons)
    n_cells = n_lat * n_lon

    if verbose:
        print(f"   Anomaly shape: {anom_arr.shape}")
        print(f"   Time range   : {times[0].date()} -> {times[-1].date()}")

    # ---- 2. Load EXP-001 extreme masks for event-based metrics --------------
    if verbose:
        print("\n[2/8] Loading EXP-001 extreme masks ...")
    has_masks = MASK_PATH.exists()
    if has_masks:
        ds_mask    = xr.open_dataset(MASK_PATH)
        ext_mask   = ds_mask["extreme_mask"].values    # (28, 125, 121) int8
        signed_mask = ds_mask["signed_extreme_mask"].values
        local_std  = ds_mask["local_std"].values       # (125, 121)
        local_thresh = ds_mask["local_threshold"].values
        ds_mask.close()
        print(f"   Loaded extreme masks from {MASK_PATH.name}")
    else:
        print("   WARNING: EXP-001 masks not found. Event-based metrics will be skipped.")
        ext_mask   = None
        local_std  = None
        local_thresh = None

    # ---- 3. Verify split time indices and build persistence pairs -----------
    if verbose:
        print("\n[3/8] Building persistence prediction pairs ...")

    # Persistence: predict(t) = anomaly(t-1)
    # For t in val_tidxs and test_tidxs, the input is anom_arr[t-1]
    # t-1 for val t=24 -> t-1=23 (May 2022, last training month)
    # t-1 for val t=25 -> t-1=24 (Jun 2022, first val month)  <- within-val, ok (t-1 is observed)
    # t-1 for test t=26 -> t-1=25 (Jul 2022, second val month) <- no leakage
    # t-1 for test t=27 -> t-1=26 (Aug 2022, first test month) <- ok

    if verbose:
        print("   Leakage check (t -> t-1 source):")
        for t in VAL_TIDXS + TEST_TIDXS:
            src = "TRAIN" if t-1 <= 23 else ("VAL" if t-1 in VAL_TIDXS else "TEST")
            split = "VAL" if t in VAL_TIDXS else "TEST"
            print(f"     {split} t={t} ({times[t].date()}) <- t-1={t-1} ({times[t-1].date()}) [{src}]")
        print("   No look-ahead leakage confirmed.")

    # Build evaluation arrays for val and test
    def get_split(tidxs):
        y_true_maps  = anom_arr[tidxs]                 # (n_months, 125, 121)
        y_pred_maps  = anom_arr[[t-1 for t in tidxs]]  # persistence
        ts_dates     = [times[t] for t in tidxs]
        ts_pred_from = [times[t-1] for t in tidxs]
        return y_true_maps, y_pred_maps, ts_dates, ts_pred_from

    val_true,  val_pred,  val_dates,  val_pred_from  = get_split(VAL_TIDXS)
    test_true, test_pred, test_dates, test_pred_from = get_split(TEST_TIDXS)

    if verbose:
        print(f"\n   Val  split: {[str(d.date()) for d in val_dates]}")
        print(f"   Test split: {[str(d.date()) for d in test_dates]}")

    # ---- 4. Compute metrics -------------------------------------------------
    if verbose:
        print("\n[4/8] Computing metrics ...")

    def compute_split_metrics(y_true_maps, y_pred_maps, dates, pred_from,
                              tidxs, split_name):
        """Compute all metrics for a given split."""
        y_true_flat = y_true_maps.ravel()
        y_pred_flat = y_pred_maps.ravel()
        n_samples   = len(y_true_flat)

        mae  = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
        rmse = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))
        r2   = safe_r2(y_true_flat, y_pred_flat)
        bias = float(np.mean(y_pred_flat) - np.mean(y_true_flat))
        corr = float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
        pred_std = float(y_pred_flat.std())
        true_std = float(y_true_flat.std())
        std_ratio = pred_std / true_std if true_std > 0 else None

        # Per-timestep stats
        timestep_stats = []
        for i, (t_idx, date, pfrom) in enumerate(zip(tidxs, dates, pred_from)):
            obs  = y_true_maps[i]
            pred = y_pred_maps[i]
            ts_mae  = float(np.mean(np.abs(obs - pred)))
            ts_rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
            ts_r2   = safe_r2(obs.ravel(), pred.ravel())
            ts_bias = float(pred.mean() - obs.mean())

            # ACC: here clim = 0 since anomaly = actual - clim already
            # So ACC simplifies to spatial Pearson correlation
            ts_acc = float(np.corrcoef(pred.ravel(), obs.ravel())[0, 1])

            # Event-based metrics
            ev = None
            if ext_mask is not None:
                obs_ext  = ext_mask[t_idx]                 # observed extreme at t
                pred_ext = ext_mask[t_idx - 1]             # predicted = previous month extreme
                ev = event_metrics(pred_ext, obs_ext)

            timestep_stats.append({
                "split"           : split_name,
                "t_idx"           : t_idx,
                "target_date"     : str(date.date()),
                "predicted_from"  : str(pfrom.date()),
                "horizon_months"  : 1,
                "MAE_mm"          : round(ts_mae,  4),
                "RMSE_mm"         : round(ts_rmse, 4),
                "R2"              : round(ts_r2, 6) if ts_r2 is not None else None,
                "bias_mm"         : round(ts_bias, 4),
                "ACC"             : round(ts_acc, 4),
                "obs_mean_mm"     : round(float(obs.mean()),  4),
                "pred_mean_mm"    : round(float(pred.mean()), 4),
                "obs_std_mm"      : round(float(obs.std()),   4),
                "pred_std_mm"     : round(float(pred.std()),  4),
                "event_metrics"   : ev,
            })

        # Aggregate event metrics across all timesteps in split
        agg_ev = None
        if ext_mask is not None:
            tp = sum(s["event_metrics"]["TP"] for s in timestep_stats)
            fp = sum(s["event_metrics"]["FP"] for s in timestep_stats)
            fn = sum(s["event_metrics"]["FN"] for s in timestep_stats)
            tn = sum(s["event_metrics"]["TN"] for s in timestep_stats)
            agg_ev = event_metrics(
                np.concatenate([ext_mask[t-1] for t in tidxs]).ravel(),
                np.concatenate([ext_mask[t]   for t in tidxs]).ravel(),
            )

        # Mean ACC across timesteps (NaN-safe)
        acc_vals = [s["ACC"] for s in timestep_stats if not np.isnan(s["ACC"])]
        mean_acc = float(np.mean(acc_vals)) if acc_vals else None

        result = {
            "split"              : split_name,
            "n_samples"          : n_samples,
            "n_timesteps"        : len(dates),
            "dates"              : [str(d.date()) for d in dates],
            "predicted_from"     : [str(d.date()) for d in pred_from],
            "MAE_mm"             : round(mae,  4),
            "RMSE_mm"            : round(rmse, 4),
            "R2"                 : round(r2, 6) if r2 is not None else None,
            "Pearson_r"          : round(corr, 4),
            "bias_mm"            : round(bias, 4),
            "actual_mean_mm"     : round(float(y_true_flat.mean()), 4),
            "predicted_mean_mm"  : round(float(y_pred_flat.mean()), 4),
            "actual_std_mm"      : round(float(y_true_flat.std()),  4),
            "predicted_std_mm"   : round(pred_std, 4),
            "std_ratio"          : round(std_ratio, 4) if std_ratio else None,
            "mean_ACC"           : round(mean_acc, 4) if mean_acc else None,
            "aggregate_event_metrics": agg_ev,
            "timestep_detail"    : timestep_stats,
            "note_R2"            : (
                "R2 < 0 means persistence is WORSE than predicting the mean. "
                "This is expected for monsoon months with high interannual variability."
            ),
            "note_ACC"           : (
                "ACC is spatial Pearson r of predicted vs observed anomaly per timestep. "
                "WMO skillful forecast threshold: ACC > 0.6."
            ),
        }
        return result

    val_metrics  = compute_split_metrics(val_true,  val_pred,  val_dates,
                                         val_pred_from,  VAL_TIDXS,  "validation")
    test_metrics = compute_split_metrics(test_true, test_pred, test_dates,
                                         test_pred_from, TEST_TIDXS, "test")

    # ---- 5. Print results ---------------------------------------------------
    if verbose:
        for m in [val_metrics, test_metrics]:
            print(f"\n  --- {m['split'].upper()} ---")
            print(f"  Samples  : {m['n_samples']:,}  ({m['n_timesteps']} months: {m['dates']})")
            print(f"  MAE      : {m['MAE_mm']} mm")
            print(f"  RMSE     : {m['RMSE_mm']} mm")
            print(f"  R2       : {m['R2']}")
            print(f"  Pearson r: {m['Pearson_r']}")
            print(f"  Bias     : {m['bias_mm']} mm")
            print(f"  Mean ACC : {m['mean_ACC']}")
            print(f"  Actual std  : {m['actual_std_mm']} mm")
            print(f"  Pred   std  : {m['predicted_std_mm']} mm")
            if m["aggregate_event_metrics"]:
                ev = m["aggregate_event_metrics"]
                print(f"  Event POD   : {ev['POD']}   FAR: {ev['FAR']}   CSI: {ev['CSI']}   ETS: {ev['ETS']}")

    # ---- 6. Save metrics JSON -----------------------------------------------
    if verbose:
        print(f"\n[5/8] Saving metrics JSON -> {METRICS_PATH.name} ...")

    output = {
        "experiment"     : "EXP-002",
        "description"    : "Persistence baseline: prediction(t) = anomaly(t-1). No training.",
        "generated_at"   : datetime.now().isoformat(),
        "methodology"    : {
            "prediction"     : "anomaly(t-1) for each grid cell",
            "horizon"        : "1 month ahead",
            "leakage"        : "None — t-1 is always a prior observed value",
            "splits"         : {
                "val" : {"t_idxs": VAL_TIDXS,  "dates": [str(d.date()) for d in val_dates]},
                "test": {"t_idxs": TEST_TIDXS, "dates": [str(d.date()) for d in test_dates]},
            },
            "note"           : (
                "Same splits as EXP-003 XGBoost. Direct comparison valid. "
                "Event-based metrics use EXP-001 extreme masks (|anomaly|>1.5*local_std)."
            ),
        },
        "validation"     : val_metrics,
        "test"           : test_metrics,
        "comparison_xgb" : {
            "note"     : "EXP-003 XGBoost baseline for reference (same splits)",
            "val_MAE"  : 59.0602,
            "val_RMSE" : 89.5691,
            "val_R2"   : -0.004286,
            "test_MAE" : 48.4801,
            "test_RMSE": 70.685,
            "test_R2"  : -0.016886,
        },
        "limitations"    : [
            "Persistence evaluated on 2-month val + 2-month test window only (small sample).",
            "Precipitation anomalies are weakly persistent; large RMSE expected.",
            "EXP-001 extreme masks derived from same 28-month record (not a robust climatology).",
            "This is a floor baseline, NOT a forecast. No model accuracy is claimed.",
        ],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"   Saved: {METRICS_PATH}")

    # ---- 7. Save predictions NetCDF -----------------------------------------
    if verbose:
        print(f"\n[6/8] Saving predictions NetCDF -> {PRED_NC_PATH.name} ...")

    all_tidxs  = VAL_TIDXS + TEST_TIDXS
    all_true   = np.concatenate([val_true,  test_true], axis=0)   # (4,125,121)
    all_pred   = np.concatenate([val_pred,  test_pred], axis=0)
    all_dates  = np.array([times[t].to_datetime64() for t in all_tidxs])
    all_split  = np.array(["val","val","test","test"])

    ds_pred = xr.Dataset(
        {
            "observed_anomaly": xr.DataArray(
                all_true.astype(np.float32),
                dims=["valid_time","latitude","longitude"],
                coords={"valid_time": all_dates, "latitude": lats, "longitude": lons},
                attrs={"long_name": "Observed monthly precipitation anomaly (mm)",
                       "units": "mm"},
            ),
            "persistence_prediction": xr.DataArray(
                all_pred.astype(np.float32),
                dims=["valid_time","latitude","longitude"],
                coords={"valid_time": all_dates, "latitude": lats, "longitude": lons},
                attrs={"long_name": "Persistence prediction = anomaly at t-1 (mm)",
                       "units": "mm",
                       "horizon": "1 month"},
            ),
            "residual": xr.DataArray(
                (all_pred - all_true).astype(np.float32),
                dims=["valid_time","latitude","longitude"],
                coords={"valid_time": all_dates, "latitude": lats, "longitude": lons},
                attrs={"long_name": "Persistence residual = prediction - observed (mm)",
                       "units": "mm"},
            ),
        },
        attrs={
            "experiment"  : "EXP-002 Persistence Baseline",
            "project"     : "VarshaVani — SIH",
            "caution"     : "This is a retrospective diagnostic, NOT a real-time forecast.",
            "created_at"  : datetime.now().isoformat(),
        },
    )
    ds_pred.to_netcdf(PRED_NC_PATH)
    print(f"   Saved: {PRED_NC_PATH}")

    # ---- 8. Verify outputs --------------------------------------------------
    if verbose:
        print("\n[7/8] Verifying outputs ...")

    # Verify JSON
    with open(METRICS_PATH) as f:
        check = json.load(f)
    assert check["experiment"] == "EXP-002", "JSON experiment key mismatch"
    assert "validation" in check and "test" in check, "JSON missing splits"
    print("   metrics JSON: PASSED")

    # Verify NetCDF
    ds_chk = xr.open_dataset(PRED_NC_PATH)
    assert ds_chk.dims["valid_time"] == 4, f"Expected 4 timesteps, got {ds_chk.dims['valid_time']}"
    assert ds_chk.dims["latitude"]   == n_lat
    assert ds_chk.dims["longitude"]  == n_lon
    nan_obs  = int(np.isnan(ds_chk["observed_anomaly"].values).sum())
    nan_pred = int(np.isnan(ds_chk["persistence_prediction"].values).sum())
    assert nan_obs  == 0, f"NaN in observed_anomaly: {nan_obs}"
    assert nan_pred == 0, f"NaN in persistence_prediction: {nan_pred}"
    # Check timestamp alignment
    loaded_times = pd.DatetimeIndex(ds_chk["valid_time"].values)
    for i, t in enumerate(all_tidxs):
        assert loaded_times[i] == times[t], f"Timestamp mismatch at index {i}"
    print(f"   NetCDF dims : {dict(ds_chk.dims)} PASSED")
    print(f"   NaN check   : obs={nan_obs}, pred={nan_pred} PASSED")
    print(f"   Timestamps  : {[str(x.date()) for x in loaded_times]} PASSED")
    ds_chk.close()
    ds_anom.close()

    # ---- 9. Visualization ---------------------------------------------------
    if verbose:
        print(f"\n[8/8] Creating visualization -> {PLOT_PATH.name} ...")
    _make_visualization(
        val_true, val_pred, val_dates,
        test_true, test_pred, test_dates,
        lats, lons, val_metrics, test_metrics
    )

    # ---- Final summary ------------------------------------------------------
    print()
    print("=" * 65)
    print("  EXP-002 COMPLETE")
    print("=" * 65)
    print(f"  Strategy : persistence (t-1 -> t), no training, no leakage")
    print(f"  Val   MAE  = {val_metrics['MAE_mm']} mm")
    print(f"  Val   RMSE = {val_metrics['RMSE_mm']} mm")
    print(f"  Val   R2   = {val_metrics['R2']}")
    print(f"  Val   ACC  = {val_metrics['mean_ACC']}")
    print(f"  Test  MAE  = {test_metrics['MAE_mm']} mm")
    print(f"  Test  RMSE = {test_metrics['RMSE_mm']} mm")
    print(f"  Test  R2   = {test_metrics['R2']}")
    print(f"  Test  ACC  = {test_metrics['mean_ACC']}")
    print()
    print("  vs XGBoost baseline (EXP-003):")
    print(f"  XGB Val  MAE=59.06 RMSE=89.57 R2=-0.0043")
    print(f"  XGB Test MAE=48.48 RMSE=70.69 R2=-0.0169")
    print()
    print("  CAUTION: Evaluated on 2-month val + 2-month test only.")
    print("  This is a skill-floor baseline, NOT a forecast.")
    print()
    print("  Outputs:")
    print(f"    {METRICS_PATH}")
    print(f"    {PRED_NC_PATH}")
    print(f"    {PLOT_PATH}")

    return output


# ---- visualization ----------------------------------------------------------

def _make_visualization(val_true, val_pred, val_dates,
                        test_true, test_pred, test_dates,
                        lats, lons, val_metrics, test_metrics):
    """
    4-panel figure:
      (a) Top row: Area-mean actual vs persistence per month (val + test)
      (b) Mid row: Spatial map — observed vs persistence for first eval month (Jun 2022)
      (c) Bottom : Scatter of all cells (actual vs persistence), residual histogram
    """
    all_true  = np.concatenate([val_true,  test_true],  axis=0)
    all_pred  = np.concatenate([val_pred,  test_pred],  axis=0)
    all_dates = val_dates + test_dates

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)

    ax_ts    = fig.add_subplot(gs[0, :])   # top: full width
    ax_mapO  = fig.add_subplot(gs[1, 0])   # mid-left: observed map
    ax_mapP  = fig.add_subplot(gs[1, 1])   # mid-right: predicted map
    ax_scat  = fig.add_subplot(gs[2, 0])   # bottom-left: scatter
    ax_hist  = fig.add_subplot(gs[2, 1])   # bottom-right: residual hist

    for ax in fig.axes:
        ax.set_facecolor("#161b22")

    # --- (a) Time series: area-mean actual vs persistence ---
    area_mean_true = [m.mean() for m in all_true]
    area_mean_pred = [m.mean() for m in all_pred]
    x = range(len(all_dates))
    date_labels = [str(d)[:7] for d in all_dates]
    split_colors = ["#60a5fa","#60a5fa","#f97316","#f97316"]  # blue=val, orange=test

    bars_true = ax_ts.bar([i - 0.2 for i in x], area_mean_true, width=0.35,
                          color="#22c55e", alpha=0.85, label="Observed (actual)")
    bars_pred = ax_ts.bar([i + 0.2 for i in x], area_mean_pred, width=0.35,
                          color="#a855f7", alpha=0.85, label="Persistence (t-1)")

    ax_ts.set_xticks(list(x))
    ax_ts.set_xticklabels(date_labels, rotation=15, ha="right", color="white", fontsize=9)
    ax_ts.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.5)
    ax_ts.axvspan(1.5, 3.5, alpha=0.08, color="#f97316", zorder=0)  # test region
    ax_ts.text(2.5, ax_ts.get_ylim()[1] if ax_ts.get_ylim()[1] != 0 else 5,
               "TEST", color="#f97316", ha="center", fontsize=9, va="bottom")
    ax_ts.set_ylabel("Area-Mean Anomaly (mm)", color="white", fontsize=10)
    ax_ts.set_title(
        f"Persistence Baseline vs Observed — Area-Mean per Month\n"
        f"Val: MAE={val_metrics['MAE_mm']} mm, RMSE={val_metrics['RMSE_mm']} mm, R2={val_metrics['R2']}\n"
        f"Test: MAE={test_metrics['MAE_mm']} mm, RMSE={test_metrics['RMSE_mm']} mm, R2={test_metrics['R2']}",
        color="white", fontsize=10)
    ax_ts.tick_params(colors="white")
    ax_ts.legend(fontsize=9, facecolor="#1c2128", labelcolor="white", framealpha=0.8)
    for spine in ax_ts.spines.values():
        spine.set_edgecolor("#444")

    # --- (b) Spatial maps: observed vs persistence for month t=0 (Jun 2022) ---
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    vmax = max(abs(all_true[0]).max(), abs(all_pred[0]).max())
    vmax = min(vmax, 300)  # cap for visual clarity

    im_o = ax_mapO.pcolormesh(lon_grid, lat_grid, all_true[0],
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im_o, ax=ax_mapO, fraction=0.046, pad=0.04).ax.tick_params(colors="white")
    ax_mapO.set_title(f"Observed Anomaly — {date_labels[0]}",
                      color="white", fontsize=10)

    im_p = ax_mapP.pcolormesh(lon_grid, lat_grid, all_pred[0],
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im_p, ax=ax_mapP, fraction=0.046, pad=0.04).ax.tick_params(colors="white")
    ax_mapP.set_title(f"Persistence Prediction — {date_labels[0]}\n(= Observed {date_labels[0]} shifted from {str(val_dates[0])[:7]})",
                      color="white", fontsize=9)

    for ax in [ax_mapO, ax_mapP]:
        ax.set_xlabel("Longitude (E)", color="white", fontsize=8)
        ax.set_ylabel("Latitude (N)", color="white", fontsize=8)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # --- (c) Scatter: all cells, all months ---
    y_true_all = all_true.ravel()
    y_pred_all = all_pred.ravel()
    rng = np.random.default_rng(42)
    sample_n = min(15000, len(y_true_all))
    idx = rng.choice(len(y_true_all), size=sample_n, replace=False)
    ax_scat.scatter(y_true_all[idx], y_pred_all[idx],
                    alpha=0.12, s=3, color="#60a5fa", rasterized=True)
    lim_max = max(abs(y_true_all).max(), abs(y_pred_all).max()) + 10
    lim_max = min(lim_max, 600)
    ax_scat.plot([-lim_max, lim_max], [-lim_max, lim_max], "r--", linewidth=1.0,
                 label="Perfect", alpha=0.8)
    ax_scat.set_xlabel("Observed anomaly (mm)", color="white", fontsize=9)
    ax_scat.set_ylabel("Persistence prediction (mm)", color="white", fontsize=9)
    combined_mae  = round(float(np.mean(np.abs(y_true_all - y_pred_all))), 2)
    combined_rmse = round(float(np.sqrt(np.mean((y_true_all - y_pred_all)**2))), 2)
    ax_scat.set_title(f"Actual vs Persistence (all eval months)\nMAE={combined_mae} mm  RMSE={combined_rmse} mm",
                      color="white", fontsize=9)
    ax_scat.legend(fontsize=8, facecolor="#1c2128", labelcolor="white")
    ax_scat.tick_params(colors="white")
    for spine in ax_scat.spines.values():
        spine.set_edgecolor("#444")

    # --- (d) Residual histogram ---
    residuals = y_pred_all - y_true_all
    ax_hist.hist(residuals, bins=80, color="#78909C", edgecolor="none")
    ax_hist.axvline(0, color="red", linewidth=1.2, linestyle="--", label="Zero error")
    ax_hist.axvline(residuals.mean(), color="#f97316", linewidth=1.2,
                    label=f"Mean bias={residuals.mean():.2f} mm")
    ax_hist.set_xlabel("Residual (persistence - observed) mm", color="white", fontsize=9)
    ax_hist.set_ylabel("Count", color="white", fontsize=9)
    ax_hist.set_title("Residual Distribution (all eval months)", color="white", fontsize=9)
    ax_hist.legend(fontsize=8, facecolor="#1c2128", labelcolor="white")
    ax_hist.tick_params(colors="white")
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#444")

    fig.suptitle(
        "EXP-002: Persistence Baseline — India ERA5 Monthly Anomaly\n"
        "Prediction(t) = Observed(t-1) | 1-month horizon | Zero free parameters | NOT a forecast",
        color="white", fontsize=11, y=1.01
    )
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Saved: {PLOT_PATH}")


# ---- entry point ------------------------------------------------------------

if __name__ == "__main__":
    result = run_exp002(verbose=True)
