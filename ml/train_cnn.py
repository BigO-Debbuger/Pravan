"""
ml/train_cnn.py
---------------
EXP-005: Training script for the CNN baseline on gridded ERA5 anomaly fields.

USAGE
-----
    python ml/train_cnn.py

INPUTS (must already exist)
----------------------------
    precipitation_anomaly.nc      (180, 125, 121) float32
    precipitation_climatology.nc  (12,  125, 121) float32

OUTPUTS (written to ml/models/exp005_cnn/)
------------------------------------------
    best_model.pt               -- best checkpoint (lowest val loss)
    final_model.pt              -- model at end of training
    config.json                 -- all hyperparameters and dataset info
    normalization_stats.json    -- anom_mean/std, clim_mean/std (train only)
    training_history.csv        -- per-epoch train/val loss and RMSE
    metrics_validation.json/.txt -- val metrics (MAE/RMSE/R² + std ratio)
    metrics_test.json/.txt       -- FINAL test metrics (run only once)
    val_spatial_diagnostic.png   -- actual/predicted/residual maps (val set)

WORKFLOW RULES ENFORCED
-----------------------
  - GPU VRAM hard ceiling: 4.0 GB. Training aborts if exceeded.
  - Full dataset stays on CPU RAM; only current batch goes to CUDA.
  - Test set is evaluated ONCE after model selection on val is complete.
  - Do NOT shuffle val or test loaders.
  - Normalization stats computed from training data ONLY.
  - Early stopping on validation loss (patience=10 epochs).
"""

import json
import pathlib
import sys
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- local imports ---------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cnn_dataset import make_dataloaders, TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END
from models.cnn_baseline import CNNBaseline, count_parameters

# ---- output directory ------------------------------------------------------
ROOT       = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "ml" / "models" / "exp005_cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- training config -------------------------------------------------------
CONFIG = {
    # Data
    "batch_size"          : 4,
    "num_workers"         : 0,
    # Model
    "in_channels"         : 4,
    "base_filters"        : 16,
    # Training
    "max_epochs"          : 100,
    "learning_rate"       : 1e-3,
    "weight_decay"        : 1e-4,
    "early_stopping_patience" : 10,
    "seed"                : 42,
    # VRAM
    "vram_ceiling_gb"     : 4.0,
    # Dataset
    "train_period"        : "2010-07 to 2020-12 (126 maps)",
    "val_period"          : "2021-01 to 2022-12 (24 maps)",
    "test_period"         : "2023-01 to 2024-12 (24 maps, held-out)",
    "input_channels"      : ["anomaly_tm3", "anomaly_tm2", "anomaly_tm1", "climatology_t"],
    "target"              : "anomaly_t (normalized; inverse-transformed for metrics)",
    "loss"                : "MSELoss",
    "experiment"          : "EXP-005 CNN Baseline",
}
VRAM_CEILING_GB = CONFIG["vram_ceiling_gb"]

# ---- VRAM helpers ----------------------------------------------------------

def get_vram_info() -> dict | None:
    """Query nvidia-smi. Returns dict or None if unavailable."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            p = [s.strip() for s in r.stdout.strip().split(",")]
            return {
                "name"     : p[0],
                "total_gb" : round(int(p[1]) / 1024, 2),
                "free_gb"  : round(int(p[2]) / 1024, 2),
                "used_gb"  : round(int(p[3]) / 1024, 2),
            }
    except Exception:
        pass
    return None


def check_vram(stage: str) -> None:
    """Print VRAM status and abort if ceiling exceeded."""
    info = get_vram_info()
    if info is None:
        print(f"   [{stage}] VRAM: nvidia-smi unavailable")
        return
    status = "OK" if info["used_gb"] < VRAM_CEILING_GB else "*** CEILING EXCEEDED ***"
    print(f"   [{stage}] VRAM: {info['used_gb']:.2f} / {info['total_gb']:.2f} GB "
          f"(free: {info['free_gb']:.2f} GB)  [{status}]")
    if info["used_gb"] >= VRAM_CEILING_GB:
        raise MemoryError(
            f"VRAM ceiling {VRAM_CEILING_GB} GB exceeded at '{stage}'. "
            f"Used: {info['used_gb']:.2f} GB. Aborting."
        )


# ---- metric helpers --------------------------------------------------------

def compute_metrics(y_true_mm: np.ndarray, y_pred_mm: np.ndarray, label: str) -> dict:
    """
    Compute MAE, RMSE, R², mean, std, and std_ratio on raw mm arrays.
    All inputs are flat numpy arrays.
    """
    mae   = float(mean_absolute_error(y_true_mm, y_pred_mm))
    rmse  = float(np.sqrt(mean_squared_error(y_true_mm, y_pred_mm)))
    r2    = float(r2_score(y_true_mm, y_pred_mm))
    a_mean = float(y_true_mm.mean())
    a_std  = float(y_true_mm.std())
    p_mean = float(y_pred_mm.mean())
    p_std  = float(y_pred_mm.std())
    ratio  = round(p_std / a_std, 4) if a_std > 0 else float("nan")

    d = {
        "label"               : label,
        "MAE_mm"              : round(mae, 4),
        "RMSE_mm"             : round(rmse, 4),
        "R2"                  : round(r2, 6),
        "actual_mean_mm"      : round(a_mean, 4),
        "actual_std_mm"       : round(a_std, 4),
        "predicted_mean_mm"   : round(p_mean, 4),
        "predicted_std_mm"    : round(p_std, 4),
        "pred_std_ratio"      : ratio,
    }

    collapse_warning = (ratio < 0.1)
    d["collapse_warning"] = collapse_warning
    return d


def metrics_to_txt(m: dict) -> str:
    w = []
    w.append(f"{'='*54}")
    w.append(f"  METRICS — {m['label']}")
    w.append(f"{'='*54}")
    w.append(f"  MAE              : {m['MAE_mm']:.4f} mm")
    w.append(f"  RMSE             : {m['RMSE_mm']:.4f} mm")
    w.append(f"  R²               : {m['R2']:.6f}")
    w.append(f"  Actual  mean/std : {m['actual_mean_mm']:.4f} / {m['actual_std_mm']:.4f} mm")
    w.append(f"  Predict mean/std : {m['predicted_mean_mm']:.4f} / {m['predicted_std_mm']:.4f} mm")
    w.append(f"  Pred/Actual std  : {m['pred_std_ratio']:.4f}"
             + (" *** COLLAPSE WARNING" if m["collapse_warning"] else ""))
    return "\n".join(w)


# ---- evaluation loop -------------------------------------------------------

@torch.no_grad()
def evaluate_loader(model, loader, device, stats, criterion):
    """Run model over a DataLoader; return loss, flat prediction/truth arrays (mm)."""
    model.eval()
    all_pred, all_true = [], []
    total_loss, n_batches = 0.0, 0

    am, as_ = stats["anom_mean"], stats["anom_std"]

    for X_cpu, y_cpu in loader:
        X = X_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        pred = model(X)
        loss = criterion(pred, y)
        total_loss += loss.item()
        n_batches  += 1

        # inverse-transform to mm for metric reporting
        pred_mm = (pred.cpu().numpy() * as_) + am
        true_mm = (y.cpu().numpy()   * as_) + am
        all_pred.append(pred_mm.ravel())
        all_true.append(true_mm.ravel())

    mean_loss = total_loss / n_batches if n_batches > 0 else float("nan")
    pred_flat = np.concatenate(all_pred)
    true_flat = np.concatenate(all_true)
    return mean_loss, pred_flat, true_flat


# ---- spatial diagnostic visualization -------------------------------------

def save_spatial_diagnostic(
    model, val_loader, device, stats, times, output_dir, split_start_idx
):
    """
    Run the model on the first validation batch; save a 3-panel spatial map:
      (1) actual anomaly, (2) predicted anomaly, (3) residual.
    Uses the FIRST validation map only (Jan 2021).
    Does NOT use test data.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("   [spatial diagnostic] matplotlib not available — skipping.")
        return

    model.eval()
    am, as_ = stats["anom_mean"], stats["anom_std"]

    with torch.no_grad():
        X_cpu, y_cpu = next(iter(val_loader))
        # Take the first sample in the batch
        X = X_cpu[[0]].to(device)
        y = y_cpu[[0]].to(device)
        pred = model(X)

    actual_mm = (y.cpu().numpy()[0, 0]    * as_) + am   # (125, 121)
    pred_mm   = (pred.cpu().numpy()[0, 0] * as_) + am   # (125, 121)
    resid_mm  = pred_mm - actual_mm

    # The first val sample is at split_start_idx in the global time array
    sample_date = str(times[split_start_idx])[:7]

    vmax = max(np.abs(actual_mm).max(), np.abs(pred_mm).max())
    vmax = float(vmax)
    cmap = "RdBu_r"
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")

    panels = [
        (actual_mm, f"Actual Anomaly\n{sample_date}"),
        (pred_mm,   f"CNN Predicted Anomaly\n{sample_date}"),
        (resid_mm,  f"Residual (Pred − Actual)\n{sample_date}"),
    ]
    for ax, (data, title) in zip(axes, panels):
        ax.set_facecolor("#1a1a2e")
        im = ax.imshow(data, cmap=cmap, norm=norm, origin="upper", aspect="auto")
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    fig.suptitle("EXP-005 CNN Baseline — Validation Spatial Diagnostic (Val Set Only)",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "val_spatial_diagnostic.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Saved: {out_path}")


# ---- main training function ------------------------------------------------

def train():
    print("=" * 60)
    print("  EXP-005: CNN BASELINE  —  ERA5 2010-2024")
    print("=" * 60)

    # Seed for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    if device.type == "cpu":
        print("  WARNING: CUDA not available. Training on CPU will be slow.")
    print()

    # Pre-training VRAM check
    check_vram("pre-training")

    # ---- 1. Data -----------------------------------------------------------
    print("\n[1/7] Building DataLoaders...")
    train_loader, val_loader, test_loader, stats, anom, times = make_dataloaders(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        normalize=True,
        verbose=True,
    )
    print(f"  Batches per epoch — Train: {len(train_loader)}, Val: {len(val_loader)}")

    # Save normalization stats (needed at inference time)
    with open(OUTPUT_DIR / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Normalization stats saved.")

    # ---- 2. Model ----------------------------------------------------------
    print("\n[2/7] Building CNN model...")
    model = CNNBaseline(
        in_channels=CONFIG["in_channels"],
        base_filters=CONFIG["base_filters"],
    ).to(device)
    n_params = count_parameters(model)
    print(f"  Architecture    : CNNBaseline(in={CONFIG['in_channels']}, base_filters={CONFIG['base_filters']})")
    print(f"  Parameters      : {n_params:,}")

    # Post-model VRAM check
    check_vram("model-loaded")

    # ---- 3. Loss / Optimizer -----------------------------------------------
    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ---- 4. Training loop --------------------------------------------------
    print(f"\n[3/7] Training (max {CONFIG['max_epochs']} epochs, "
          f"early_stop patience={CONFIG['early_stopping_patience']})...")
    print(f"  Loss            : MSELoss (normalized space)")
    print(f"  LR              : {CONFIG['learning_rate']}")
    print(f"  Batch size      : {CONFIG['batch_size']}")
    print()

    history = []
    best_val_loss  = float("inf")
    patience_count = 0
    best_epoch     = -1

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        epoch_t0 = time.time()

        # --- train ---
        model.train()
        train_loss_sum = 0.0
        for X_cpu, y_cpu in train_loader:
            X = X_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        train_loss = train_loss_sum / len(train_loader)

        # --- validate ---
        val_loss, _, _ = evaluate_loader(model, val_loader, device, stats, criterion)

        elapsed = time.time() - epoch_t0
        history.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "val_loss"  : round(val_loss, 6),
            "train_rmse": round(float(np.sqrt(train_loss)) * stats["anom_std"], 4),
            "val_rmse"  : round(float(np.sqrt(val_loss))   * stats["anom_std"], 4),
            "elapsed_s" : round(elapsed, 2),
        })

        # Print every 5 epochs or on improvement
        improved = val_loss < best_val_loss
        if epoch % 5 == 0 or improved or epoch == 1:
            marker = " ← best" if improved else ""
            print(f"  Epoch {epoch:>3d}/{CONFIG['max_epochs']}  "
                  f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                  f"val_RMSE={np.sqrt(val_loss)*stats['anom_std']:.2f}mm  "
                  f"({elapsed:.1f}s){marker}")

        # Save best checkpoint
        if improved:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : val_loss,
                "stats"      : stats,
                "config"     : CONFIG,
            }, OUTPUT_DIR / "best_model.pt")
        else:
            patience_count += 1

        # VRAM check every 10 epochs
        if epoch % 10 == 0:
            check_vram(f"epoch-{epoch}")

        # Early stopping
        if patience_count >= CONFIG["early_stopping_patience"]:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {patience_count} epochs). "
                  f"Best epoch: {best_epoch}")
            break

    # Save final model
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "val_loss"   : val_loss,
        "stats"      : stats,
        "config"     : CONFIG,
    }, OUTPUT_DIR / "final_model.pt")

    # Save training history
    import csv
    with open(OUTPUT_DIR / "training_history.csv", "w", newline="") as f:
        if history:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    print(f"\n  Training complete. Best epoch: {best_epoch}, best_val_loss={best_val_loss:.6f}")

    # ---- 5. Load best model for evaluation ---------------------------------
    print(f"\n[4/7] Loading best model (epoch {best_epoch}) for evaluation...")
    ckpt = torch.load(OUTPUT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    check_vram("post-training")

    # ---- 6. Validation metrics (on best model) -----------------------------
    print("\n[5/7] Computing validation metrics...")
    _, val_pred_mm, val_true_mm = evaluate_loader(model, val_loader, device, stats, criterion)
    val_metrics = compute_metrics(val_true_mm, val_pred_mm, "VALIDATION (Jan 2021 - Dec 2022)")
    val_metrics["best_epoch"] = best_epoch
    val_metrics["best_val_loss_normalized"] = round(best_val_loss, 6)
    val_metrics["xgb_baseline_val_rmse_mm"] = 73.48   # EXP-003 reference
    val_metrics["xgb_baseline_val_r2"]       = -0.0034

    print(metrics_to_txt(val_metrics))
    with open(OUTPUT_DIR / "metrics_validation.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(OUTPUT_DIR / "metrics_validation.txt", "w") as f:
        f.write(metrics_to_txt(val_metrics))

    # ---- 7. Spatial diagnostic (val set only) ------------------------------
    print("\n[6/7] Generating spatial diagnostic (validation set, first month)...")
    save_spatial_diagnostic(
        model, val_loader, device, stats, times, OUTPUT_DIR, VAL_START
    )

    # ---- 8. Test metrics (run only once, after model selection) ------------
    print("\n[7/7] Final test-set evaluation (held-out, one-time run)...")
    _, test_pred_mm, test_true_mm = evaluate_loader(model, test_loader, device, stats, criterion)
    test_metrics = compute_metrics(test_true_mm, test_pred_mm, "TEST (Jan 2023 - Dec 2024, held-out)")
    test_metrics["best_epoch"] = best_epoch
    test_metrics["xgb_baseline_test_rmse_mm"] = 69.50
    test_metrics["xgb_baseline_test_r2"]       = 0.0002
    test_metrics["scientific_note"] = (
        "Test set evaluated once on the best validation checkpoint. "
        "Test data was never used during training, early stopping, or normalization. "
        "CNN vs XGBoost comparison: see xgb_baseline_* fields."
    )

    print(metrics_to_txt(test_metrics))
    with open(OUTPUT_DIR / "metrics_test.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(OUTPUT_DIR / "metrics_test.txt", "w") as f:
        f.write(metrics_to_txt(test_metrics))

    # ---- Save config -------------------------------------------------------
    CONFIG["actual_epochs_trained"] = epoch
    CONFIG["best_epoch"]            = best_epoch
    CONFIG["n_model_parameters"]    = n_params
    CONFIG["device"]                = str(device)
    vram_final = get_vram_info()
    CONFIG["final_vram_used_gb"]    = vram_final["used_gb"] if vram_final else None
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EXP-005 COMPLETE")
    print("=" * 60)
    print(f"  Val  RMSE : {val_metrics['RMSE_mm']:.2f} mm   "
          f"(XGBoost baseline: 73.48 mm)")
    print(f"  Val  R²   : {val_metrics['R2']:.4f}   "
          f"(XGBoost baseline: -0.0034)")
    print(f"  Test RMSE : {test_metrics['RMSE_mm']:.2f} mm   "
          f"(XGBoost baseline: 69.50 mm)")
    print(f"  Test R²   : {test_metrics['R2']:.4f}   "
          f"(XGBoost baseline: 0.0002)")
    if val_metrics["collapse_warning"]:
        print("  *** COLLAPSE WARNING: predicted std << actual std on validation set")
    else:
        print(f"  Pred/Actual std ratio (val) : {val_metrics['pred_std_ratio']:.3f}")
    print(f"\n  Output dir : {OUTPUT_DIR}")
    if vram_final:
        print(f"  Final VRAM : {vram_final['used_gb']:.2f} / {vram_final['total_gb']:.2f} GB"
              f"  (ceiling: {VRAM_CEILING_GB:.1f} GB)")
    return val_metrics, test_metrics


if __name__ == "__main__":
    train()
