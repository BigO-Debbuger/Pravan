"""
ml/cnn_dataset.py
-----------------
PyTorch Dataset for EXP-005: CNN on Gridded ERA5 Anomaly Fields.

DESIGN
------
- Loads precipitation_anomaly.nc (180, 125, 121) and
  precipitation_climatology.nc (12, 125, 121) entirely into CPU RAM.
  Total size: ~11 MB. The GPU never sees the full arrays.
- Each sample (t) yields:
    X = [anomaly(t-3), anomaly(t-2), anomaly(t-1), climatology(month_t)]
        shape: (4, 125, 121), float32
    y = anomaly(t)
        shape: (1, 125, 121), float32
- Normalization statistics are computed ONLY from training-period inputs.
  Val and Test use the same (train) statistics — no leakage.

TIME INDEX CONVENTIONS (verified 2026-09-25):
  Index 0  = Jan 2010
  Index 6  = Jul 2010  ← first training sample (needs t-3 = Apr 2010 = idx 3)
  Index 131 = Dec 2020  ← last  training sample
  Index 132 = Jan 2021  ← first validation sample
  Index 155 = Dec 2022  ← last  validation sample
  Index 156 = Jan 2023  ← first test sample
  Index 179 = Dec 2024  ← last  test sample

NORMALIZATION
-------------
  anomaly channels : z = (x - ANOM_MEAN) / ANOM_STD
      computed from anomaly[3:132]  (all lag values visible during training)
  climatology channel : z = (x - CLIM_MEAN) / CLIM_STD
      computed from climatology[:] (only 12 months, no temporal leakage)
  target (anomaly at t) : same ANOM_MEAN / ANOM_STD
      → inverse-transform to mm for metric reporting

SPLITS
------
  TRAIN_INDICES = range(6, 132)   → 126 maps
  VAL_INDICES   = range(132, 156) → 24  maps
  TEST_INDICES  = range(156, 180) → 24  maps
"""

import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
ROOT     = pathlib.Path(__file__).resolve().parent.parent
ANOM_NC  = ROOT / "precipitation_anomaly.nc"
CLIM_NC  = ROOT / "precipitation_climatology.nc"

# Split boundaries (integer indices into the 180-step time axis)
TRAIN_START = 6    # Jul 2010  (needs t-3=3 which is Apr 2010, within the array)
TRAIN_END   = 132  # exclusive → last sample is idx 131 = Dec 2020
VAL_START   = 132  # Jan 2021
VAL_END     = 156  # exclusive → last sample is idx 155 = Dec 2022
TEST_START  = 156  # Jan 2023
TEST_END    = 180  # exclusive → last sample is idx 179 = Dec 2024

# ---------------------------------------------------------------------------

def load_arrays():
    """
    Load anomaly and climatology arrays from NetCDF into CPU numpy arrays.
    Called once at startup; does NOT touch GPU.
    Returns:
        anom  : np.ndarray (180, 125, 121) float32 — monthly anomaly
        clim  : np.ndarray (12,  125, 121) float32 — monthly climatology
        times : np.ndarray (180,)           — datetime64 time axis
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required. Install with: pip install xarray netCDF4")

    if not ANOM_NC.exists():
        raise FileNotFoundError(f"Anomaly NetCDF not found: {ANOM_NC}")
    if not CLIM_NC.exists():
        raise FileNotFoundError(f"Climatology NetCDF not found: {CLIM_NC}")

    ds_anom = xr.open_dataset(ANOM_NC)
    ds_clim = xr.open_dataset(CLIM_NC)

    anom  = ds_anom["tp"].values.astype(np.float32)   # (180, 125, 121)
    clim  = ds_clim["tp"].values.astype(np.float32)   # (12,  125, 121)
    times = ds_anom["valid_time"].values               # (180,) datetime64

    ds_anom.close()
    ds_clim.close()

    assert anom.shape  == (180, 125, 121), f"Unexpected anom shape: {anom.shape}"
    assert clim.shape  == (12,  125, 121), f"Unexpected clim shape: {clim.shape}"
    assert not np.isnan(anom).any(),  "NaNs detected in anomaly array"
    assert not np.isnan(clim).any(),  "NaNs detected in climatology array"

    return anom, clim, times


def compute_normalization_stats(anom: np.ndarray, clim: np.ndarray) -> dict:
    """
    Compute normalization statistics strictly from training-period data.
    - Anomaly stats: from anom[3:132]  (indices 3–131 cover every lag-3..lag-1
      value seen during training; does NOT touch val/test indices 132-179).
    - Climatology stats: from clim[:]  (12 fixed monthly maps, no time leakage).

    Returns a dict with keys:
        anom_mean, anom_std, clim_mean, clim_std
    """
    anom_train = anom[3:TRAIN_END]  # (129, 125, 121)
    stats = {
        "anom_mean": float(np.mean(anom_train)),
        "anom_std":  float(np.std(anom_train)),
        "clim_mean": float(np.mean(clim)),
        "clim_std":  float(np.std(clim)),
    }
    assert stats["anom_std"] > 0, "anom_std is zero — data problem"
    assert stats["clim_std"] > 0, "clim_std is zero — data problem"
    return stats


class ERA5AnomalyCNNDataset(Dataset):
    """
    PyTorch Dataset for EXP-005.

    Parameters
    ----------
    anom        : np.ndarray (180, 125, 121) float32
    clim        : np.ndarray (12,  125, 121) float32
    times       : np.ndarray (180,) datetime64
    time_indices: sequence of ints — which time steps this split covers
    stats       : dict from compute_normalization_stats() — MUST be from Train only
    normalize   : bool — if False, return raw values (useful for sanity checks)
    """

    def __init__(
        self,
        anom: np.ndarray,
        clim: np.ndarray,
        times: np.ndarray,
        time_indices,
        stats: dict,
        normalize: bool = True,
    ):
        self.anom    = anom
        self.clim    = clim
        self.times   = times
        self.indices = list(time_indices)
        self.stats   = stats
        self.normalize = normalize

        # Sanity: every requested index must have t-3 available (idx >= 3)
        assert all(i >= 3 for i in self.indices), \
            "Some time indices do not have enough history for a 3-month lag."
        assert all(i < 180 for i in self.indices), \
            "Some time indices exceed the array length (180)."

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        """
        Returns:
            X : torch.Tensor (4, 125, 121) — 3 lag anomaly channels + climatology
            y : torch.Tensor (1, 125, 121) — target anomaly at time t
        """
        t = self.indices[item]

        # Extract calendar month for climatology indexing (0-based)
        # times[t] is datetime64; extract month via string slicing (safe, portable)
        month_str = str(self.times[t])[:7]   # "YYYY-MM"
        month_0   = int(month_str.split("-")[1]) - 1  # 0-indexed: Jan→0, Dec→11

        # ---- Build input channels ------------------------------------------
        anom_tm3 = self.anom[t - 3]   # (125, 121)
        anom_tm2 = self.anom[t - 2]
        anom_tm1 = self.anom[t - 1]
        clim_t   = self.clim[month_0] # (125, 121)

        # ---- Build target --------------------------------------------------
        target = self.anom[t]          # (125, 121)

        # Stack to (4, 125, 121) and (1, 125, 121)
        X = np.stack([anom_tm3, anom_tm2, anom_tm1, clim_t], axis=0)  # (4, 125, 121)
        y = target[np.newaxis, :, :]                                    # (1, 125, 121)

        # ---- Normalize (Z-score, training stats) ---------------------------
        if self.normalize:
            am, as_ = self.stats["anom_mean"], self.stats["anom_std"]
            cm, cs  = self.stats["clim_mean"], self.stats["clim_std"]
            X[0] = (X[0] - am) / as_   # anomaly t-3
            X[1] = (X[1] - am) / as_   # anomaly t-2
            X[2] = (X[2] - am) / as_   # anomaly t-1
            X[3] = (X[3] - cm) / cs    # climatology channel
            y    = (y    - am) / as_   # target anomaly

        return torch.from_numpy(X), torch.from_numpy(y)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Convert normalized target values back to raw millimeters.
        y_scaled can be any shape.
        """
        am = self.stats["anom_mean"]
        as_ = self.stats["anom_std"]
        return y_scaled * as_ + am


# ---------------------------------------------------------------------------
# Factory functions

def make_dataloaders(
    batch_size: int = 4,
    num_workers: int = 0,
    normalize: bool = True,
    verbose: bool = True,
):
    """
    Load data, compute normalization, and return train/val/test DataLoaders.

    The test DataLoader is NOT shuffled and is intended for final evaluation only.

    Returns
    -------
    (train_loader, val_loader, test_loader, stats, anom, times)
    stats : dict — normalization parameters (save these alongside the model)
    anom  : full anomaly array (CPU) — used for inverse-transform
    times : time axis — for reporting dates
    """
    anom, clim, times = load_arrays()

    if verbose:
        print(f"  Anomaly array  : {anom.shape} float32  ({anom.nbytes / 1e6:.1f} MB)")
        print(f"  Climatology    : {clim.shape} float32  ({clim.nbytes / 1e6:.1f} MB)")
        print(f"  Total CPU RAM  : {(anom.nbytes + clim.nbytes) / 1e6:.1f} MB")

    # Compute normalization stats from TRAINING period ONLY
    stats = compute_normalization_stats(anom, clim)

    if verbose:
        print(f"  Normalization stats (training only, no leakage):")
        print(f"    anomaly  : mean={stats['anom_mean']:.4f}, std={stats['anom_std']:.4f}")
        print(f"    climatol : mean={stats['clim_mean']:.4f}, std={stats['clim_std']:.4f}")

    train_idx = range(TRAIN_START, TRAIN_END)  # 126 samples
    val_idx   = range(VAL_START,   VAL_END)    # 24  samples
    test_idx  = range(TEST_START,  TEST_END)   # 24  samples

    ds_train = ERA5AnomalyCNNDataset(anom, clim, times, train_idx, stats, normalize)
    ds_val   = ERA5AnomalyCNNDataset(anom, clim, times, val_idx,   stats, normalize)
    ds_test  = ERA5AnomalyCNNDataset(anom, clim, times, test_idx,  stats, normalize)

    if verbose:
        print(f"  Train samples  : {len(ds_train)} maps")
        print(f"  Val samples    : {len(ds_val)} maps")
        print(f"  Test samples   : {len(ds_test)} maps  (held-out)")

    loader_kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, stats, anom, times


if __name__ == "__main__":
    # Quick self-test: verify one sample from each split
    print("Running cnn_dataset.py self-test...")
    tl, vl, tsl, stats, anom, times = make_dataloaders(batch_size=4, verbose=True)
    for name, loader in [("Train", tl), ("Val", vl), ("Test", tsl)]:
        X, y = next(iter(loader))
        print(f"  [{name}] X: {tuple(X.shape)}, y: {tuple(y.shape)}, "
              f"X dtype: {X.dtype}, NaN in X: {X.isnan().any()}, NaN in y: {y.isnan().any()}")
    print("Self-test complete.")
