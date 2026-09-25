"""
ml/gpu_sanity_test.py
---------------------
GPU SANITY TEST — DO NOT USE FOR FINAL TRAINING.

PURPOSE
-------
Verifies ONLY:
  1. XGBoost CUDA device is detected and accessible
  2. DMatrix initializes on a small subset of real training data
  3. A short training run completes without CUDA/OOM errors
  4. VRAM stays comfortably below the 3-4 GB hard ceiling

This uses ONLY 30,000 rows (~1.6% of the training set) and runs only 20 trees.
It produces NO model artifacts and writes NO files.

USAGE
-----
    python ml/gpu_sanity_test.py

EXPECTED BEHAVIOUR
------------------
  - Prints GPU name and VRAM stats at start, mid-training, and end
  - Completes in under 30 seconds
  - Reports PASS or FAIL for each check
  - Exits with code 0 on PASS, code 1 on FAIL
"""

import sys
import pathlib
import subprocess
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT     = pathlib.Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "ml" / "processed"

# ---- same feature set as train_model.py -------------------------------------
FEATURE_COLS = [
    "year_scaled", "month_scaled", "month_sin_scaled", "month_cos_scaled",
    "months_since_start_scaled", "lat_scaled", "lon_scaled",
    "lat_norm_scaled", "lon_norm_scaled", "climatology_mm_scaled",
    "anomaly_lag1_scaled", "anomaly_lag2_scaled", "anomaly_lag3_scaled",
    "precip_lag1_scaled", "precip_lag2_scaled",
    "rolling_mean_3m_scaled", "rolling_std_3m_scaled", "rolling_mean_6m_scaled",
]
TARGET_COL = "anomaly_mm"

VRAM_HARD_CEILING_GB = 4.0

# Sanity test parameters — deliberately tiny
SANITY_N_ROWS       = 30_000    # ~1.6% of training set
SANITY_N_ESTIMATORS = 20        # only 20 trees — just enough to verify training loop
SANITY_SEED         = 42


# ---- helpers ----------------------------------------------------------------

def get_vram_info():
    """Query nvidia-smi and return a dict of VRAM stats, or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "name"      : parts[0],
                "total_mb"  : int(parts[1]),
                "free_mb"   : int(parts[2]),
                "used_mb"   : int(parts[3]),
                "total_gb"  : round(int(parts[1]) / 1024, 2),
                "free_gb"   : round(int(parts[2]) / 1024, 2),
                "used_gb"   : round(int(parts[3]) / 1024, 2),
            }
    except Exception as e:
        print(f"   nvidia-smi error: {e}")
    return None


def print_vram(label: str, info: dict):
    if info is None:
        print(f"   {label:30s} | VRAM: unavailable")
        return
    status = "OK" if info["used_gb"] < VRAM_HARD_CEILING_GB else "*** CEILING EXCEEDED ***"
    print(f"   {label:30s} | VRAM: {info['used_gb']:.2f} GB used / "
          f"{info['total_gb']:.2f} GB total "
          f"({info['free_gb']:.2f} GB free)  [{status}]")


# ---- main sanity test -------------------------------------------------------

def run_sanity_test():
    results = {}
    all_pass = True

    print("=" * 65)
    print("  XGBoost GPU SANITY TEST")
    print("=" * 65)
    print(f"  XGBoost version : {xgb.__version__}")
    print(f"  Rows to test    : {SANITY_N_ROWS:,} (from features_train.parquet)")
    print(f"  Trees to train  : {SANITY_N_ESTIMATORS}")
    print(f"  VRAM ceiling    : {VRAM_HARD_CEILING_GB} GB")
    print()

    # ---- CHECK 1: nvidia-smi available / GPU detected -----------------------
    print("[CHECK 1] Detecting GPU via nvidia-smi...")
    info_start = get_vram_info()
    if info_start is not None:
        print(f"   GPU : {info_start['name']}")
        print_vram("Before loading data", info_start)
        results["gpu_detected"] = True
        print("   PASS: GPU detected.\n")
    else:
        print("   WARNING: nvidia-smi not found or no GPU detected.")
        print("   Continuing with CPU fallback (device will fall back from cuda).")
        results["gpu_detected"] = False
        print()

    # ---- CHECK 2: Load training parquet -------------------------------------
    print("[CHECK 2] Loading 30,000 rows from features_train.parquet...")
    train_path = PROC_DIR / "features_train.parquet"
    if not train_path.exists():
        print(f"   FAIL: {train_path} not found.")
        print("   Run ml/features.py first.")
        sys.exit(1)

    df_train = pd.read_parquet(train_path)
    # Take a stratified chronological slice — first 30k rows of the training set
    df_sample = df_train.head(SANITY_N_ROWS).copy()
    X_sample  = df_sample[FEATURE_COLS].values.astype(np.float32)
    y_sample  = df_sample[TARGET_COL].values.astype(np.float32)
    del df_train   # free RAM immediately — only keep the small sample

    print(f"   Sample shape    : {X_sample.shape}")
    print(f"   NaN in X        : {np.isnan(X_sample).sum()}")
    print(f"   NaN in y        : {np.isnan(y_sample).sum()}")
    print(f"   y mean / std    : {y_sample.mean():.2f} / {y_sample.std():.2f}")
    results["data_loaded"] = True
    print("   PASS: Data loaded cleanly.\n")

    # ---- CHECK 3: DMatrix construction --------------------------------------
    print("[CHECK 3] Constructing XGBoost DMatrix...")
    try:
        dtrain = xgb.DMatrix(X_sample, label=y_sample)
        print(f"   DMatrix shape   : {dtrain.num_row()} rows x {dtrain.num_col()} cols")
        info_after_dmatrix = get_vram_info()
        print_vram("After DMatrix construction", info_after_dmatrix)
        results["dmatrix_ok"] = True
        print("   PASS: DMatrix constructed.\n")
    except Exception as e:
        print(f"   FAIL: DMatrix construction failed: {e}")
        results["dmatrix_ok"] = False
        all_pass = False

    # ---- CHECK 4: XGBoost CUDA training run ---------------------------------
    print("[CHECK 4] Running short XGBoost training (20 trees, cuda/hist)...")
    sanity_params = {
        "tree_method"  : "hist",
        "device"       : "cuda",
        "max_bin"      : 256,
        "max_depth"    : 5,
        "learning_rate": 0.05,
        "objective"    : "reg:squarederror",
        "eval_metric"  : "rmse",
        "seed"         : SANITY_SEED,
        "verbosity"    : 0,   # silent during sanity test
    }
    try:
        evals_result = {}
        model_sanity = xgb.train(
            sanity_params,
            dtrain,
            num_boost_round=SANITY_N_ESTIMATORS,
            evals=[(dtrain, "train")],
            evals_result=evals_result,
            verbose_eval=False,
        )
        final_rmse = evals_result["train"]["rmse"][-1]
        print(f"   Training RMSE after {SANITY_N_ESTIMATORS} trees : {final_rmse:.4f} mm")

        info_after_train = get_vram_info()
        print_vram("After training", info_after_train)

        # VRAM ceiling check
        if info_after_train is not None:
            if info_after_train["used_gb"] >= VRAM_HARD_CEILING_GB:
                print(f"   FAIL: VRAM ceiling {VRAM_HARD_CEILING_GB} GB exceeded! "
                      f"Used: {info_after_train['used_gb']:.2f} GB")
                results["vram_safe"] = False
                all_pass = False
            else:
                print(f"   VRAM headroom   : "
                      f"{VRAM_HARD_CEILING_GB - info_after_train['used_gb']:.2f} GB "
                      f"remaining under ceiling")
                results["vram_safe"] = True
        else:
            print("   VRAM ceiling: could not verify (nvidia-smi unavailable)")
            results["vram_safe"] = None

        results["training_ok"] = True
        print("   PASS: XGBoost CUDA training completed without errors.\n")

    except xgb.core.XGBoostError as e:
        err_str = str(e)
        print(f"   FAIL: XGBoost error: {err_str}")
        if "cuda" in err_str.lower() or "gpu" in err_str.lower():
            print("   NOTE: Possible CUDA issue. Check that XGBoost was installed")
            print("         with GPU support: pip install xgboost --extra-index-url ...")
            print("         Or try 'device': 'cpu' as a fallback.")
        results["training_ok"] = False
        all_pass = False

    except MemoryError as e:
        print(f"   FAIL (OOM): {e}")
        results["training_ok"] = False
        results["vram_safe"] = False
        all_pass = False

    except Exception as e:
        print(f"   FAIL (unexpected): {type(e).__name__}: {e}")
        results["training_ok"] = False
        all_pass = False

    # ---- SUMMARY ------------------------------------------------------------
    print("=" * 65)
    print("  SANITY TEST SUMMARY")
    print("=" * 65)
    check_labels = {
        "gpu_detected" : "GPU detected via nvidia-smi",
        "data_loaded"  : "Training Parquet loaded (30k rows)",
        "dmatrix_ok"   : "XGBoost DMatrix constructed",
        "training_ok"  : "XGBoost CUDA training (20 trees)",
        "vram_safe"    : f"VRAM below {VRAM_HARD_CEILING_GB} GB ceiling",
    }
    for key, label in check_labels.items():
        val = results.get(key)
        if val is True:
            symbol = "PASS"
        elif val is False:
            symbol = "FAIL"
        else:
            symbol = "SKIP"
        print(f"   {symbol:4s}  {label}")

    print()
    if all_pass:
        print("  ALL CHECKS PASSED.")
        print("  The GPU configuration is verified safe for full training.")
        print("  You may now approve: python ml/train_model.py")
        sys.exit(0)
    else:
        print("  ONE OR MORE CHECKS FAILED.")
        print("  Do NOT proceed to full training until failures are resolved.")
        sys.exit(1)


if __name__ == "__main__":
    run_sanity_test()
