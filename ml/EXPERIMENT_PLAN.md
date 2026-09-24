# EXPERIMENT_PLAN.md
# VarshaVani — ML Experiment Plan
## AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies

> Last Updated: 2026-09-24
> Status: Planned — no new experiments run yet. EXP-003 is the only completed experiment.

---

## OVERVIEW

This plan defines a staged progression of experiments, ordered by model complexity.
Each experiment builds on the previous. A more complex experiment is only pursued if
the data volume and compute requirements are satisfied.

**Guiding Principle:**
> Never skip to a complex model to look impressive.
> Each baseline anchors the interpretation of the next.

---

## EXP-001: Statistical Extreme-Event Baseline

**Status:** NOT STARTED

### Goal
Detect extreme precipitation anomalies using only statistical thresholds.
No machine learning. No training loop.

### Input
- `precipitation_anomaly.nc` (28 × 125 × 121)
- For each grid cell: compute mean and standard deviation over all 28 months

### Target
- Binary label: **is this (month, cell) an extreme event?**
- Threshold: `|anomaly| > 1.5σ` (local per-cell σ from the available record)

### Prediction Horizon
- None (retrospective labeling, not forecasting)

### Train / Val / Test Strategy
- No split required — this is a threshold classifier, not a trained model
- Evaluate coverage: what fraction of cells are labeled as extreme in Jun-Sep vs Oct-May

### Metrics
| Metric | Rationale |
|---|---|
| Fraction of cells labeled extreme per month | Sanity check: expect higher fraction in monsoon months |
| Spatial distribution of extreme events | Do events cluster in high-rainfall regions? |
| False alarm rate vs 2021 monsoon ground truth | If we had a verification dataset |

**Extreme-event metrics (to use in all experiments):**
| Metric | Symbol | Description |
|---|---|---|
| Probability of Detection | POD | Fraction of observed extremes correctly predicted |
| False Alarm Rate | FAR | Fraction of predicted extremes that were not observed |
| Critical Success Index | CSI | POD / (POD + FAR + Misses) — rewards correct detections |
| Equitable Threat Score | ETS | CSI adjusted for random chance |
| Fractions Skill Score | FSS | Spatial tolerance metric for binary maps |

### Expected Output
- `ml/experiments/exp001_extreme_masks.nc` — binary extreme mask per (month, lat, lon)
- `ml/experiments/exp001_stats.json` — per-month event counts and spatial statistics

### Expected Limitations
- σ estimated from only 28 months (2 monsoon seasons). Thresholds are not climatologically robust.
- No temporal dependence: every timestep treated independently.

---

## EXP-002: Persistence Baseline

**Status:** NOT STARTED

### Goal
Predict next month's anomaly by simply repeating the current month's anomaly.
This is the simplest possible dynamic forecast and is the standard floor for evaluating any ML model.

> If a model cannot beat persistence, it is useless as a forecasting tool.

### Input
- `anomaly_lag1` (anomaly at month t-1) for each (t, lat, lon)
- This column already exists in the parquet files

### Target
- `anomaly_mm` at month t

### Prediction Horizon
- 1 month ahead (t-1 → t)

### Train / Val / Test Strategy
- No training required
- Evaluate on the val and test splits already defined:
  - Val: Jun–Jul 2022
  - Test: Aug–Sep 2022

### Metrics
| Metric | Expected Value |
|---|---|
| MAE | ~50–80 mm (persistence is poor for monsoon onset) |
| RMSE | ~80–120 mm |
| R² | Likely negative (worse than climatology mean for monsoon months) |
| Anomaly Correlation Coefficient (ACC) | 0.0–0.2 |

**Anomaly Correlation Coefficient:**
```
ACC = Σ[(pred - clim)(obs - clim)] / sqrt(Σ(pred-clim)² × Σ(obs-clim)²)
```
Standard NWP verification metric. ACC > 0.6 is the WMO threshold for "skillful" forecasts.

### Expected Output
- `ml/experiments/exp002_persistence_metrics.json`

### Expected Limitations
- Precipitation anomalies are weakly persistent month-to-month (especially at monsoon onset/offset).
- Persistence will perform worst in Jun and Sep (transition months).

---

## EXP-003: XGBoost Baseline (COMPLETED)

**Status:** ✅ COMPLETE

### Summary of Results
| Metric | Val (Jun-Jul 2022) | Test (Aug-Sep 2022) |
|---|---|---|
| MAE | 59.06 mm | 48.48 mm |
| RMSE | 89.57 mm | 70.69 mm |
| R² | -0.0043 | -0.0169 |
| Pred std / Actual std | 0.014 | 0.045 |

### Diagnosis
- **Mean collapse**: Model produces near-constant predictions (std ratio 0.014).
- **Root cause**: Only 18 training months, one monsoon season. Insufficient for learning interannual variability.
- **Spatial treatment**: Each (cell, month) is an independent sample — spatial neighborhood information is ignored.
- **Best stopping iteration**: 0 (early stopping triggered immediately — val RMSE rose from iteration 0 onward).
- **Top features**: `month` (42.8%), `month_sin` (24.4%), `anomaly_lag3` (13.6%). Temporal cyclic features dominate because the model reduces to a seasonal mean.

### Model Artifacts
- `ml/models/xgb_baseline.json`
- `ml/models/model_config.json`
- `ml/models/metrics_validation.json` / `.txt`
- `ml/models/metrics_test.json` / `.txt`
- `ml/models/feature_importance.csv` / `.png`
- `ml/models/validation_predictions.png`

---

## EXP-004: Spatial Anomaly Tracking (Object-Based Analysis)

**Status:** NOT STARTED

### Goal
Move from per-cell prediction to **event-level** tracking.
Detect contiguous anomaly blobs, assign event IDs, and track them through time.

This is not a forecasting experiment — it is a **detection + tracking** experiment
that transforms the point-prediction output into physically meaningful events.

### Input
- `precipitation_anomaly.nc` (28 × 125 × 121) — existing
- Extreme event masks from EXP-001

### Target
Not a scalar regression target. Instead:
- **Event objects**: connected regions where `|anomaly| > threshold`
- **Event tracks**: associations between blobs at t and t+1

### Algorithm
1. Threshold anomaly field → binary mask
2. Label connected components (scipy.ndimage.label)
3. Compute per-event:
   - centroid lat/lon
   - total area (km²)
   - mean intensity (mm)
   - max intensity (mm)
   - bounding box
4. Match events across timesteps (centroid proximity threshold)
5. Assign persistent Event IDs

### Output Format
```json
{
  "event_id": "EVT-2021-001",
  "birth_time": "2021-06-30",
  "death_time": "2021-09-30",
  "track": [
    {"time": "2021-06-30", "centroid_lat": 15.5, "centroid_lon": 75.2, "area_km2": 145000, "intensity_mm": 180.0},
    {"time": "2021-07-31", "centroid_lat": 16.1, "centroid_lon": 75.8, "area_km2": 160000, "intensity_mm": 220.0},
    ...
  ],
  "max_intensity": 220.0,
  "persistence_months": 4,
  "severity_score": 0.87
}
```

### Metrics
| Metric | Description |
|---|---|
| Event count per season | Sanity check against known 2021 monsoon |
| Average event area | km² |
| Average event persistence | Months |
| Spatial overlap between events at t and t+1 | Tracking quality |

### Expected Limitations
- Monthly resolution means events span full months, losing sub-monthly dynamics.
- With only 28 months, the "event catalog" will be very small (~10–20 events total).

---

## EXP-005: CNN on Gridded Anomaly Fields (Conditional on Data Extension)

**Status:** NOT STARTED — BLOCKED pending extended ERA5 dataset

**Prerequisite:** ERA5 extended to at least 2010–2024 (~168 usable months)

### Goal
Train a 2D CNN to learn spatial anomaly patterns from gridded input fields.
Each input is a 2D anomaly map; output is a predicted anomaly map 1 month ahead.

### Architecture Sketch
```
Input: Stack of k lag anomaly maps  [k × 125 × 121]
       + Climatology map             [1 × 125 × 121]
       → Total input: (k+1) × 125 × 121

Conv2D(32, 3×3) → ReLU → BatchNorm
Conv2D(64, 3×3) → ReLU → BatchNorm
Conv2D(64, 3×3) → ReLU → BatchNorm
Conv2D(1, 1×1)  → Linear activation

Output: [1 × 125 × 121] — predicted anomaly map at t+1
```

### Input
- Stack of 3 consecutive monthly anomaly maps as separate channels
- Climatology for the target month

### Target
- Full anomaly map at month t+1 (continuous regression per cell)

### Prediction Horizon
- 1 month ahead

### Train / Val / Test Strategy
- Chronological split: first 80% of months → train, next 10% → val, last 10% → test
- Minimum: 100 training maps, 15 val maps, 15 test maps

### Metrics
| Metric | Description |
|---|---|
| Spatial RMSE | RMSE across all 125×121 cells |
| Spatial ACC | Anomaly Correlation Coefficient (spatial pattern skill) |
| FSS | Fractions Skill Score at multiple neighborhood sizes |
| Extreme event recall | CSI for cells with |anomaly| > 1.5σ |

### Expected Limitations
- 2D spatial convolution ignores temporal evolution. ConvLSTM needed for sequence learning.
- Indian monsoon has strong synoptic-to-mesoscale structure; 0.25° monthly data is coarse.

---

## EXP-006: ConvLSTM (Conditional on Data Extension)

**Status:** NOT STARTED — BLOCKED pending extended ERA5 dataset

**Prerequisite:** ERA5 extended to 1990–2024 (~400+ usable months), GPU compute

### Goal
Model spatiotemporal precipitation anomaly dynamics using ConvLSTM.
The model processes a sequence of 2D anomaly maps and predicts the next map.

### Architecture Sketch
```
Input sequence: [T, C, H, W] = [12 months, C_channels, 125, 121]

ConvLSTM2D(hidden_dim=64, kernel_size=3) → hidden state sequence
ConvLSTM2D(hidden_dim=64, kernel_size=3) → hidden state sequence
Conv2D(1, kernel_size=1) → point-wise projection

Output: [1, 125, 121] — next-month anomaly map
```

### Physical Motivation
- ConvLSTM's recurrent spatial state naturally encodes "memory" of previous anomaly patterns
- Spatial convolution preserves neighborhood structure discarded by tabular XGBoost
- Multi-step rollout enables 2–3 month lead predictions

### Data Requirement
- 300+ training maps for reliable convolutional filter learning
- GPU with at least 8 GB VRAM for batched spatiotemporal training

### Expected Limitations
- Even ConvLSTM cannot compensate for missing atmospheric driver variables
- Overfitting likely without SST and wind boundary conditions

---

## METRIC REFERENCE TABLE

### Regression Metrics
| Metric | Formula | Good Value |
|---|---|---|
| MAE | mean(|pred - obs|) | < climatology MAE |
| RMSE | sqrt(mean((pred-obs)²)) | < persistence RMSE |
| R² | 1 - SS_res/SS_tot | > 0 (positive skill) |
| ACC | see above | > 0.6 (WMO skillful threshold) |

### Extreme-Event Metrics
| Metric | Formula | Good Value |
|---|---|---|
| POD (recall) | TP / (TP + FN) | > 0.5 |
| FAR | FP / (FP + TP) | < 0.4 |
| CSI | TP / (TP + FP + FN) | > 0.3 |
| ETS | (CSI - random) / (1 - random) | > 0.1 |
| FSS | 1 - MSE_f / MSE_ref | > 0.5 at neighborhood N |

### Spatial Metrics
| Metric | Description |
|---|---|
| Bias | mean(pred) - mean(obs) across all cells |
| Spatial correlation | Pearson r of predicted map vs observed map |
| Pattern RMSE | RMSE computed spatially per timestep |

---

## PROGRESS TRACKER

| Experiment | Status | Blocks |
|---|---|---|
| EXP-001: Statistical extreme baseline | NOT STARTED | Nothing |
| EXP-002: Persistence baseline | NOT STARTED | Nothing |
| EXP-003: XGBoost baseline | ✅ COMPLETE | — |
| EXP-004: Spatial event tracking | NOT STARTED | EXP-001 output |
| EXP-005: CNN on gridded fields | NOT STARTED | Extended ERA5 needed |
| EXP-006: ConvLSTM | NOT STARTED | Extended ERA5 + GPU needed |

**Immediate next step: Implement EXP-001 (statistical extreme labeling) and EXP-002 (persistence baseline).**
These require no new data and no new model training.
