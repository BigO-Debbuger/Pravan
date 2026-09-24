"""
ml/train_model.py
-----------------
Trains an XGBoost regression baseline to predict monthly precipitation anomaly (mm).

USAGE
-----
    python ml/train_model.py

INPUTS (must already exist)
-----------------------------
    ml/processed/features_train.parquet
    ml/processed/features_val.parquet
    ml/processed/features_test.parquet

OUTPUTS
-------
    ml/models/xgb_baseline.json        -- trained XGBoost model (booster format)
    ml/models/model_config.json        -- hyperparameters used
    ml/models/metrics_validation.json  -- MAE, RMSE, R2 on val set
    ml/models/metrics_validation.txt   -- human-readable val metrics
    ml/models/metrics_test.json        -- FINAL MAE, RMSE, R2 on test set
    ml/models/metrics_test.txt         -- human-readable test metrics
    ml/models/feature_importance.csv   -- XGBoost feature importance (gain)
    ml/models/feature_importance.png   -- bar chart of top-18 features
    ml/models/validation_predictions.png -- actual vs predicted scatter + time plot

SCIENTIFIC LIMITATIONS (explicitly documented)
----------------------------------------------
  - The training dataset has only 28 monthly time steps (22 after NaN lag drop).
    This is a very short climatological record. Any metrics should be interpreted
    with caution. The model may not generalize well to unseen weather regimes or
    years outside the 2020-2022 training window.
  - Non-monsoon months (Oct-May) have near-zero rainfall at most grid cells.
    The model will learn this trivially. Monsoon months (Jun-Sep) are the
    scientifically interesting and harder-to-predict regime.
  - Do NOT claim strong forecasting ability from a single 28-month training run.
"""

import pathlib
import json
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- paths -----------------------------------------------------------------
ROOT      = pathlib.Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "ml" / "processed"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- feature columns (18 raw-scaled features, same order as features.py) ---
FEATURE_COLS = [
    "year_scaled",
    "month_scaled",
    "month_sin_scaled",
    "month_cos_scaled",
    "months_since_start_scaled",
    "lat_scaled",
    "lon_scaled",
    "lat_norm_scaled",
    "lon_norm_scaled",
    "climatology_mm_scaled",
    "anomaly_lag1_scaled",
    "anomaly_lag2_scaled",
    "anomaly_lag3_scaled",
    "precip_lag1_scaled",
    "precip_lag2_scaled",
    "rolling_mean_3m_scaled",
    "rolling_std_3m_scaled",
    "rolling_mean_6m_scaled",
]

# Friendly display names for plots
FEATURE_DISPLAY = [
    "year",
    "month",
    "month_sin",
    "month_cos",
    "months_since_start",
    "lat",
    "lon",
    "lat_norm",
    "lon_norm",
    "climatology_mm",
    "anomaly_lag1",
    "anomaly_lag2",
    "anomaly_lag3",
    "precip_lag1",
    "precip_lag2",
    "rolling_mean_3m",
    "rolling_std_3m",
    "rolling_mean_6m",
]

TARGET_COL = "anomaly_mm"


# ---- XGBoost hyperparameters -----------------------------------------------
#
# Rationale for each parameter:
#   n_estimators=400   : Enough trees for a tabular dataset; not so many that
#                        the model memorises the limited 22 training months.
#   max_depth=5        : Moderate depth. Deep trees (>6) overfit badly on small
#                        temporal datasets. Shallow trees (<=3) underfit.
#   learning_rate=0.05 : Conservative. Slow learning rate + more trees generally
#                        beats fast learning rate + few trees.
#   subsample=0.8      : Stochastic gradient boosting. Reduces variance.
#   colsample_bytree=0.8: Random feature subsampling per tree. Prevents any
#                         single dominant feature from monopolising every tree.
#   min_child_weight=10: Minimum sum of instance weight in a leaf. Higher value
#                        regularises against overfitting on rare grid cells.
#   gamma=0.1          : Minimum loss reduction to make a split. Small positive
#                        value adds soft regularisation.
#   reg_lambda=1.0     : L2 regularisation (ridge). XGBoost default.
#   reg_alpha=0.0      : L1 regularisation (lasso). Off by default.
#   objective=reg:squarederror : Standard MSE regression loss.
#   eval_metric=rmse   : RMSE on val set used for early stopping.
#   early_stopping_rounds=30 : Stop if val RMSE does not improve for 30 rounds.
#                              Prevents overfitting to the small dataset.
#   seed=42            : Reproducibility.
#
# These are "sensible baseline" parameters, NOT the result of a hyperparameter
# search. Tuning was intentionally not performed because:
#   (a) the dataset is too small for reliable cross-validation tuning
#   (b) the goal is a documented, reproducible baseline, not peak performance.

XGB_PARAMS = {
    "n_estimators"       : 400,
    "max_depth"          : 5,
    "learning_rate"      : 0.05,
    "subsample"          : 0.8,
    "colsample_bytree"   : 0.8,
    "min_child_weight"   : 10,
    "gamma"              : 0.1,
    "reg_lambda"         : 1.0,
    "reg_alpha"          : 0.0,
    "objective"          : "reg:squarederror",
    "eval_metric"        : "rmse",
    "early_stopping_rounds": 30,
    "seed"               : 42,
    "verbosity"          : 1,
}


# ---- helpers ----------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    metrics = {
        "split"               : label,
        "n_samples"           : int(len(y_true)),
        "MAE_mm"              : round(mae,  4),
        "RMSE_mm"             : round(rmse, 4),
        "R2"                  : round(r2,   6),
        "actual_mean_mm"      : round(float(y_true.mean()), 4),
        "predicted_mean_mm"   : round(float(y_pred.mean()), 4),
        "actual_std_mm"       : round(float(y_true.std()),  4),
        "predicted_std_mm"    : round(float(y_pred.std()),  4),
        "predicted_min_mm"    : round(float(y_pred.min()),  4),
        "predicted_max_mm"    : round(float(y_pred.max()),  4),
    }
    return metrics


def metrics_to_txt(metrics: dict, extra_notes: str = "") -> str:
    lines = [
        "=" * 60,
        f"  METRICS — {metrics['split'].upper()}",
        "=" * 60,
        f"  Samples          : {metrics['n_samples']:,}",
        f"  MAE              : {metrics['MAE_mm']:.4f} mm",
        f"  RMSE             : {metrics['RMSE_mm']:.4f} mm",
        f"  R2               : {metrics['R2']:.6f}",
        "",
        f"  Actual mean      : {metrics['actual_mean_mm']:.4f} mm",
        f"  Predicted mean   : {metrics['predicted_mean_mm']:.4f} mm",
        f"  Actual std       : {metrics['actual_std_mm']:.4f} mm",
        f"  Predicted std    : {metrics['predicted_std_mm']:.4f} mm",
        f"  Predicted min    : {metrics['predicted_min_mm']:.4f} mm",
        f"  Predicted max    : {metrics['predicted_max_mm']:.4f} mm",
    ]
    if extra_notes:
        lines += ["", "  NOTES:", extra_notes]
    lines += ["=" * 60]
    return "\n".join(lines)


# ---- main training function -------------------------------------------------

def train():
    print("=" * 60)
    print("  XGBoost BASELINE TRAINING")
    print("=" * 60)
    print(f"  XGBoost version : {xgb.__version__}")
    print(f"  Features        : {len(FEATURE_COLS)}")
    print(f"  Target          : {TARGET_COL}")

    # ---- 1. Load data -------------------------------------------------------
    print("\n[1/6] Loading parquet files...")
    df_train = pd.read_parquet(PROC_DIR / "features_train.parquet")
    df_val   = pd.read_parquet(PROC_DIR / "features_val.parquet")
    df_test  = pd.read_parquet(PROC_DIR / "features_test.parquet")

    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    y_train = df_train[TARGET_COL].values.astype(np.float32)
    X_val   = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val   = df_val[TARGET_COL].values.astype(np.float32)
    X_test  = df_test[FEATURE_COLS].values.astype(np.float32)
    y_test  = df_test[TARGET_COL].values.astype(np.float32)

    print(f"   Train : {X_train.shape[0]:,} rows")
    print(f"   Val   : {X_val.shape[0]:,} rows")
    print(f"   Test  : {X_test.shape[0]:,} rows  [HELD OUT - not used during tuning]")

    # ---- 2. Sanity checks before training -----------------------------------
    print("\n[2/6] Pre-training sanity checks...")
    for name, X, y in [("Train", X_train, y_train),
                        ("Val",   X_val,   y_val),
                        ("Test",  X_test,  y_test)]:
        nan_X = np.isnan(X).sum()
        nan_y = np.isnan(y).sum()
        if nan_X > 0 or nan_y > 0:
            print(f"   WARNING: {name} has {nan_X} NaN in features, {nan_y} NaN in target")
        else:
            print(f"   {name}: no NaN detected. OK.")

    # Distribution check
    print(f"\n   Train target  mean={y_train.mean():.2f}  std={y_train.std():.2f}"
          f"  min={y_train.min():.1f}  max={y_train.max():.1f}")
    print(f"   Val   target  mean={y_val.mean():.2f}  std={y_val.std():.2f}"
          f"  min={y_val.min():.1f}  max={y_val.max():.1f}")
    print(f"   Test  target  mean={y_test.mean():.2f}  std={y_test.std():.2f}"
          f"  min={y_test.min():.1f}  max={y_test.max():.1f}")

    # NOTE: Large std difference between train and test is expected.
    # Val/test cover monsoon months (Jun-Sep), which have much higher variance
    # than the non-monsoon months that dominate the training set.

    # ---- 3. Train XGBoost ---------------------------------------------------
    print("\n[3/6] Training XGBoost regressor...")
    print(f"   Parameters: {XGB_PARAMS}")

    # Separate early_stopping_rounds from the constructor params
    # (XGBoost 2.x: pass in constructor for best practice)
    params_for_init = {k: v for k, v in XGB_PARAMS.items()
                       if k != "early_stopping_rounds"}
    # Add early_stopping_rounds to constructor (XGBoost 2.x preferred style)
    params_for_init["early_stopping_rounds"] = XGB_PARAMS["early_stopping_rounds"]
    early_stop = None   # not passed to fit() to avoid deprecation warning

    model = xgb.XGBRegressor(**params_for_init)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"\n   Best iteration (early stopping): {best_iter}")

    # ---- 4. Save model + config ---------------------------------------------
    print("\n[4/6] Saving model and config...")
    model.save_model(MODEL_DIR / "xgb_baseline.json")
    print(f"   Model saved to: {MODEL_DIR / 'xgb_baseline.json'}")

    config = {
        "model_type"         : "XGBRegressor",
        "xgboost_version"    : xgb.__version__,
        "best_iteration"     : int(best_iter),
        "n_features"         : len(FEATURE_COLS),
        "feature_cols"       : FEATURE_COLS,
        "feature_display"    : FEATURE_DISPLAY,
        "target_col"         : TARGET_COL,
        "hyperparameters"    : XGB_PARAMS,
        "train_rows"         : int(X_train.shape[0]),
        "val_rows"           : int(X_val.shape[0]),
        "test_rows"          : int(X_test.shape[0]),
        "dataset_note"       : (
            "Training dataset covers only ~22 effective monthly timesteps "
            "(28 months minus 6 dropped for lag window). This is a very small "
            "climatological sample. Metrics should be interpreted cautiously. "
            "The model may not generalize to years/weather regimes not represented "
            "in the 2020-2022 ERA5 training window."
        ),
    }
    with open(MODEL_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- 5. Validation metrics ----------------------------------------------
    print("\n[5/6] Evaluating on VALIDATION set...")
    y_val_pred = model.predict(X_val)

    val_metrics = compute_metrics(y_val, y_val_pred, "validation")

    # Leakage / collapse checks
    pred_std  = float(y_val_pred.std())
    true_std  = float(y_val.std())
    std_ratio = pred_std / true_std if true_std > 0 else 0
    mean_bias = float(y_val_pred.mean()) - float(y_val.mean())

    val_metrics["predicted_std_ratio_to_actual"] = round(std_ratio, 4)
    val_metrics["mean_bias_mm"] = round(mean_bias, 4)

    warnings = []
    if val_metrics["R2"] > 0.99:
        warnings.append("WARNING: R2 > 0.99 is suspiciously high. Investigate leakage.")
    if std_ratio < 0.3:
        warnings.append(f"WARNING: Predictions may be collapsing toward mean "
                        f"(pred_std/true_std = {std_ratio:.3f}).")
    if abs(mean_bias) > 10:
        warnings.append(f"WARNING: Large mean bias of {mean_bias:.2f} mm detected.")
    if not warnings:
        warnings.append("No obvious collapse, extreme bias, or suspiciously high R2 detected.")

    val_metrics["scientific_limitation"] = (
        "Dataset spans 28 months (22 usable after lag window drops). Val set covers "
        "Jun-Jul 2022 (2 early monsoon months with active rainfall variance). "
        "Because training contains only one prior monsoon season (2021), interannual "
        "variability is high and generalization is strictly constrained by sample size."
    )

    print(metrics_to_txt(val_metrics))
    for w in warnings:
        print(f"   {w}")

    with open(MODEL_DIR / "metrics_validation.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(MODEL_DIR / "metrics_validation.txt", "w") as f:
        txt = metrics_to_txt(val_metrics)
        txt += "\n\nSCIENTIFIC CHECKS:\n"
        for w in warnings:
            txt += f"  {w}\n"
        txt += f"\nSCIENTIFIC LIMITATION:\n  {val_metrics['scientific_limitation']}\n"
        f.write(txt)

    # ---- 6. Feature importance ----------------------------------------------
    print("\n[6/6] Computing and saving feature importance...")
    importance = model.get_booster().get_score(importance_type="gain")

    # Map internal feature names (f0, f1, ...) to display names
    fi_dict = {}
    for k, v in importance.items():
        idx = int(k.replace("f", ""))
        fi_dict[FEATURE_DISPLAY[idx]] = v

    fi_df = pd.DataFrame(
        list(fi_dict.items()), columns=["feature", "importance_gain"]
    ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    # Features not used at all get 0 importance
    used_features = set(fi_df["feature"])
    for feat in FEATURE_DISPLAY:
        if feat not in used_features:
            fi_df = pd.concat([fi_df, pd.DataFrame(
                [{"feature": feat, "importance_gain": 0.0}]
            )], ignore_index=True)

    fi_df = fi_df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    fi_df["rank"] = fi_df.index + 1
    fi_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    print(f"   Saved: {MODEL_DIR / 'feature_importance.csv'}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3" if i < 5 else "#90CAF9" for i in range(len(fi_df))]
    bars = ax.barh(fi_df["feature"][::-1], fi_df["importance_gain"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title("XGBoost Feature Importance — ERA5 Precipitation Anomaly Model",
                 fontsize=13, pad=12)
    ax.axvline(x=0, color="black", linewidth=0.8)
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + max(fi_df["importance_gain"]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}", va="center", fontsize=8, color="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {MODEL_DIR / 'feature_importance.png'}")

    top3 = fi_df.head(3)["feature"].tolist()
    print(f"   Top-3 features by gain: {top3}")

    # ---- 7. Validation prediction visualization -----------------------------
    print("\nGenerating validation prediction visualization...")

    # Get val time indices and dates
    val_time_idxs = df_val["time_idx"].astype(int).values
    val_months    = df_val["month"].astype(int).values
    val_lats      = df_val["lat"].values
    val_lons      = df_val["lon"].values

    # For visualization: aggregate spatial mean per time step
    val_df_vis = pd.DataFrame({
        "time_idx"  : val_time_idxs,
        "y_true"    : y_val,
        "y_pred"    : y_val_pred,
        "lat"       : val_lats,
        "lon"       : val_lons,
    })

    # Spatial mean per month
    monthly_vis = val_df_vis.groupby("time_idx").agg(
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
        y_true_std =("y_true", "std"),
        y_pred_std =("y_pred", "std"),
    ).reset_index()

    # Load time index -> date mapping from the anomaly file
    import xarray as xr
    anom_ds = xr.open_dataset(ROOT / "precipitation_anomaly.nc")
    times_all = pd.DatetimeIndex(anom_ds.valid_time.values)

    monthly_vis["date"] = monthly_vis["time_idx"].apply(
        lambda i: str(times_all[i])[:7]
    )

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # top: full width
    ax2 = fig.add_subplot(gs[1, 0])   # bottom-left
    ax3 = fig.add_subplot(gs[1, 1])   # bottom-right

    # (a) Monthly area-mean: actual vs predicted
    x  = range(len(monthly_vis))
    ax1.bar([i - 0.2 for i in x], monthly_vis["y_true_mean"],
            width=0.4, color="#1976D2", alpha=0.85, label="Actual anomaly (area mean)")
    ax1.bar([i + 0.2 for i in x], monthly_vis["y_pred_mean"],
            width=0.4, color="#F57C00", alpha=0.85, label="Predicted anomaly (area mean)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(monthly_vis["date"], rotation=30, ha="right")
    ax1.set_ylabel("Anomaly (mm)", fontsize=11)
    ax1.set_title("Validation Set: Area-Mean Actual vs Predicted Anomaly per Month",
                  fontsize=12)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.legend(fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (b) Scatter: all grid cells
    sample_n = min(10000, len(y_val))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_val), size=sample_n, replace=False)
    ax2.scatter(y_val[idx], y_val_pred[idx],
                alpha=0.15, s=4, color="#1976D2", rasterized=True)
    lims = [min(y_val.min(), y_val_pred.min()) - 10,
            max(y_val.max(), y_val_pred.max()) + 10]
    ax2.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
    ax2.set_xlabel("Actual anomaly (mm)", fontsize=11)
    ax2.set_ylabel("Predicted anomaly (mm)", fontsize=11)
    ax2.set_title(f"Actual vs Predicted (n={sample_n:,} cells sampled)\n"
                  f"MAE={val_metrics['MAE_mm']:.2f} mm  RMSE={val_metrics['RMSE_mm']:.2f} mm"
                  f"  R2={val_metrics['R2']:.4f}",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # (c) Residual distribution
    residuals = y_val_pred - y_val
    ax3.hist(residuals, bins=60, color="#78909C", edgecolor="white", linewidth=0.3)
    ax3.axvline(0, color="red", linewidth=1.2, linestyle="--", label="Zero error")
    ax3.axvline(residuals.mean(), color="orange", linewidth=1.2,
                linestyle="-", label=f"Mean bias={residuals.mean():.2f} mm")
    ax3.set_xlabel("Residual (predicted - actual) mm", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Residual Distribution (Validation Set)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.suptitle("XGBoost Baseline — Validation Set Performance\n"
                 "(ERA5 Monthly Precipitation Anomaly, India 2020-2022)",
                 fontsize=13, y=1.01)
    plt.savefig(MODEL_DIR / "validation_predictions.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {MODEL_DIR / 'validation_predictions.png'}")

    # ---- 8. FINAL TEST evaluation (held-out, run only once) ----------------
    print("\n" + "=" * 60)
    print("  FINAL TEST SET EVALUATION (run only once, held-out)")
    print("=" * 60)
    y_test_pred = model.predict(X_test)

    test_metrics = compute_metrics(y_test, y_test_pred, "test (FINAL)")
    test_pred_std  = float(y_test_pred.std())
    test_true_std  = float(y_test.std())
    test_std_ratio = test_pred_std / test_true_std if test_true_std > 0 else 0
    test_mean_bias = float(y_test_pred.mean()) - float(y_test.mean())

    test_metrics["predicted_std_ratio_to_actual"] = round(test_std_ratio, 4)
    test_metrics["mean_bias_mm"] = round(test_mean_bias, 4)
    test_metrics["scientific_limitation"] = (
        "Test set covers Aug-Sep 2022 (2 held-out late monsoon months). This represents "
        "a strict out-of-time evaluation during peak monsoon variability. "
        "With only 2 months of test snapshots, metrics should be interpreted with awareness "
        "of short-record sample variance."
    )

    test_warnings = []
    if test_metrics["R2"] > 0.99:
        test_warnings.append("WARNING: R2 > 0.99 is suspiciously high.")
    if test_std_ratio < 0.3:
        test_warnings.append(f"WARNING: Predictions collapsing toward mean "
                             f"(std ratio = {test_std_ratio:.3f}).")
    if not test_warnings:
        test_warnings.append("No obvious collapse or extreme bias detected on test set.")
    test_metrics["scientific_checks"] = test_warnings

    print(metrics_to_txt(test_metrics))
    for w in test_warnings:
        print(f"   {w}")

    with open(MODEL_DIR / "metrics_test.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    txt = metrics_to_txt(test_metrics)
    txt += "\n\nFINAL TEST SET - HELD OUT\n"
    txt += "Test covers Aug-Sep 2022 (2 held-out late monsoon months)\n\n"
    txt += "SCIENTIFIC CHECKS:\n"
    for w in test_warnings:
        txt += f"  {w}\n"
    txt += f"\nSCIENTIFIC LIMITATION:\n  {test_metrics['scientific_limitation']}\n"
    with open(MODEL_DIR / "metrics_test.txt", "w") as f:
        f.write(txt)

    # ---- 9. Verify all output files exist -----------------------------------
    print("\n" + "=" * 60)
    print("  OUTPUT FILES VERIFICATION")
    print("=" * 60)
    expected = [
        "xgb_baseline.json",
        "model_config.json",
        "metrics_validation.json",
        "metrics_validation.txt",
        "metrics_test.json",
        "metrics_test.txt",
        "feature_importance.csv",
        "feature_importance.png",
        "validation_predictions.png",
    ]
    all_ok = True
    for fname in expected:
        fpath = MODEL_DIR / fname
        exists = fpath.exists()
        size   = fpath.stat().st_size if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"   {status}  {fname}  ({size:,} bytes)")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n   All expected output files confirmed present.")
    else:
        print("\n   WARNING: Some expected files are missing!")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Val  MAE  = {val_metrics['MAE_mm']:.2f} mm")
    print(f"  Val  RMSE = {val_metrics['RMSE_mm']:.2f} mm")
    print(f"  Val  R2   = {val_metrics['R2']:.4f}")
    print(f"  Test MAE  = {test_metrics['MAE_mm']:.2f} mm")
    print(f"  Test RMSE = {test_metrics['RMSE_mm']:.2f} mm")
    print(f"  Test R2   = {test_metrics['R2']:.4f}")
    print(f"  Top features: {fi_df.head(5)['feature'].tolist()}")
    print()
    print("  IMPORTANT LIMITATION:")
    print("  The training dataset covers only ~22 effective monthly timesteps.")
    print("  These metrics should NOT be used to claim reliable real-world")
    print("  forecasting ability. They represent in-sample performance on a")
    print("  very short ERA5 record (Jun 2020 - Sep 2022).")

    return {
        "val_metrics" : val_metrics,
        "test_metrics": test_metrics,
        "fi_df"       : fi_df,
    }


# ---- entry point ------------------------------------------------------------

if __name__ == "__main__":
    results = train()
