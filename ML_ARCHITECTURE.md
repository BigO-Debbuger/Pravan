# ML_ARCHITECTURE.md
# AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies
## VarshaVani — SIH ML Architecture

> Last Updated: 2026-09-24
> Status: Architecture design phase — no new models trained yet.

---

## 1. AUDIT FINDINGS: CURRENT ML STATE

### 1.1 What Data We Have

| Item | Detail |
|---|---|
| Raw Source | ERA5 hourly total precipitation (m/hr), India domain, Jun 2020 – Sep 2022 |
| Temporal resolution | Aggregated to monthly totals (28 monthly snapshots) |
| Spatial resolution | 0.25° × 0.25°, 125 lat × 121 lon = 15,125 grid cells |
| Derived files | `precipitation_climatology.nc` (12 × 125 × 121), `precipitation_anomaly.nc` (28 × 125 × 121) |
| Available for ML | 22 usable months after 6-month lag window drop |

### 1.2 Current Features (18 total)

**Group: Temporal (5)**
- `year`, `month`, `month_sin`, `month_cos`, `months_since_start`

**Group: Spatial (4)**
- `lat`, `lon`, `lat_norm`, `lon_norm`

**Group: Climatology (1)**
- `climatology_mm` — 12-month mean at that cell

**Group: Lag Anomaly (3)**
- `anomaly_lag1`, `anomaly_lag2`, `anomaly_lag3`

**Group: Lag Rainfall and Rolling (5)**
- `precip_lag1`, `precip_lag2`, `rolling_mean_3m`, `rolling_std_3m`, `rolling_mean_6m`

### 1.3 Target Variable

- `anomaly_mm` — continuous precipitation anomaly in mm at each grid cell for the current month.
- Rationale: regression preserves magnitude information; classification requires arbitrary thresholds not defensible with <3 years of data.

### 1.4 Train / Val / Test Split

| Split | Months | Date Range | Rows | Notes |
|---|---|---|---|---|
| Train | 18 | Dec 2020 – May 2022 | 272,250 | Contains one monsoon season (2021) |
| Val | 2 | Jun – Jul 2022 | 30,250 | Early 2022 monsoon |
| Test | 2 | Aug – Sep 2022 | 30,250 | Held-out late monsoon |

### 1.5 Leakage Assessment

**Temporal Leakage:**
- ✅ No temporal leakage. All lag features are strictly lookback (t-1, t-2, t-3). Current month's total rainfall is excluded from features.
- ✅ Train/val/test is a strictly chronological split with no shuffle.
- ✅ StandardScaler fit on train only and applied to val/test.
- ⚠️ MINOR: The climatology baseline is computed from the full 28-month record, not just the training window. This is a minor issue — climatology represents long-term seasonal patterns, not future observations, and is acceptable scientifically. However, it should be noted for future reference.

**Spatial Leakage:**
- ✅ No spatial leakage. Spatial coordinates (lat, lon) are static and not data-derived.
- ✅ Lag features are computed independently per grid cell; no cross-cell contamination.
- ✅ Each row is one (grid cell × month) sample; rows are independent at the feature level.

### 1.6 Lag Feature Validity

- ✅ `anomaly_lag1/2/3` use strict lookback (`t-k` indexing), verified in `features.py` lines 163–203.
- ✅ `rolling_mean_3m` uses window `arr[t-3:t]` — excludes current month. Valid.
- ✅ `rolling_mean_6m` uses window `arr[t-6:t]` — excludes current month. Valid.
- ✅ First 6 months dropped to ensure no NaN rolling windows exist in training data.

### 1.7 Why the XGBoost Result Was Poor

**Root causes in descending order of impact:**

1. **Sample size is fundamentally insufficient.** With 18 training months (1 monsoon season), the model has seen exactly one Indian monsoon cycle. It cannot distinguish interannual variability (ENSO, IOD, heat-low interactions) from noise. XGBoost needed ~5–10 monsoon years minimum for stable tree splits.

2. **Prediction std / actual std = 0.014 on val.** The model outputs a nearly constant prediction (≈ regional mean) regardless of input. This is "mean collapse" — the optimal strategy for a model that cannot explain variance. With R² ≈ −0.004, the model performs only marginally above a simple mean prediction.

3. **Spatially independent rows discard neighborhood context.** The current tabular format treats each (cell, month) pair as an independent sample. A grid cell in Kerala gets no information about what happened in adjacent cells or upstream. This discards critical spatial structure in precipitation patterns.

4. **No atmospheric dynamics features.** We have only precipitation. No wind, no moisture flux, no OLR, no SST, no CAPE. These are the physical drivers of anomalies. Without them, even a perfect model has bounded predictability.

5. **Early stopping at iteration 0.** The model found no improvement at all on validation set, indicating severe generalization failure on even the "easiest" 2-month window.

### 1.8 Is the Current Monthly Dataset Sufficient for ML?

**For tabular/regression ML (XGBoost, RF):** No. 18 training months provides too few temporal samples for reliable learning. Minimum recommended: 60+ months.

**For CNN on spatial fields:** Marginal. 22 usable monthly snapshots = 22 input maps. Too few for deep CNNs without heavy augmentation or transfer learning.

**For ConvLSTM:** No. ConvLSTM needs long temporal sequences to learn recurrent spatial dynamics. 22 timesteps is insufficient.

**For statistical baselines (climatology, persistence):** Yes. These require no learning.

**For feature-based spatial clustering / anomaly detection:** Yes, the spatial resolution is excellent for unsupervised anomaly region identification.

### 1.9 What Additional Data / Resolution Is Needed

**Priority 1 — More years of ERA5 precipitation:**
- Extend record from 2020–2022 to at least 1979–2024 (~500+ months)
- ERA5 is publicly available; download cost ~10 GB for full India domain

**Priority 2 — Atmospheric driver variables (future experiments, do NOT download now):**
- `u10`, `v10` (10m wind components) — monsoon flow patterns
- `tcwv` (total column water vapour) — moisture availability
- `sst` (sea surface temperature, Bay of Bengal / Arabian Sea) — ocean-atmosphere coupling for ENSO/IOD
- `cape` (convective available potential energy) — convective instability
- `z500` (500 hPa geopotential height) — large-scale steering flow
- `olr` (outgoing longwave radiation) — cloud/convection proxy (NOAA AVHRR or ERA5)

**Priority 3 — Sub-monthly resolution:**
- 6-hourly or daily aggregates would allow medium-range (3–10 day) forecasting
- Currently limited to monthly resolution only

---

## 2. ML ARCHITECTURE DESIGN

### 2.1 Design Philosophy

> **"Match model complexity to data volume."**
>
> Do not use a deep learning model because it sounds impressive. Use the simplest model that can represent the physics of the problem given the available data. Graduate complexity only when data volume justifies it.

The architecture is designed as a **staged progression**:
```
Stage 1: Statistical + Persistence baselines  (no learning required)
Stage 2: XGBoost (tabular, already done)
Stage 3: Spatial clustering / anomaly detection (unsupervised)
Stage 4: CNN on gridded fields (if extended ERA5 record available)
Stage 5: ConvLSTM / spatiotemporal model (if 5+ years available)
Stage 6: Medium-range forecasting (if sub-monthly data available)
```

### 2.2 Task Decomposition

The full problem is decomposed into these sub-tasks:

#### A. Extreme Anomaly Detection
**Goal:** Flag grid cells where precipitation anomaly exceeds a meaningful threshold.
- Threshold: |anomaly| > 1.5σ (local standard deviation, computed per cell)
- Method: Purely statistical; no ML model required for this step
- Output: Binary mask (extreme / normal) per (month, lat, lon)

#### B. Spatio-Temporal Feature Extraction
**Goal:** Extract spatial structures from 2D anomaly fields per timestep.
- Method: Thresholded binary masks → connected component labeling (scipy.ndimage)
- Output: List of connected blobs per timestep (event candidates)
- No ML required for detection; ML improves intensity estimation

#### C. Extreme Event Prediction
**Goal:** Predict whether an extreme precipitation event will occur at a location in the next 1–4 weeks.
- Required data: Sub-monthly resolution ERA5 (daily or 6-hourly) + atmospheric drivers
- Model: ConvLSTM or CNN+LSTM if enough data; XGBoost if only monthly
- Current limitation: Monthly data only → prediction horizon = 1 month ahead

#### D. Spatial Event / Object Detection
**Goal:** Identify and label contiguous anomaly regions as distinct events.
- Method: OpenCV or scipy connected-component analysis on thresholded anomaly maps
- Each event gets a unique label, bounding box, and centroid
- No ML model required; purely algorithmic

#### E. Event Tracking Through Time
**Goal:** Link event blobs across consecutive timesteps.
- Method: Minimum centroid distance matching (Hungarian algorithm or nearest-centroid)
- Assign Event ID, birth time, death time
- Track:
  ```
  Event #001
  t0 → centroid (lat=15.2, lon=75.4)
  t1 → centroid (lat=16.1, lon=76.2)
  t2 → centroid (lat=17.0, lon=77.0)
  ```

#### F. Event Severity Estimation
For each tracked event, compute:
| Metric | Computation |
|---|---|
| **Intensity** | Mean anomaly (mm) within the event blob |
| **Area** | Number of grid cells × cell area (km²) |
| **Centroid** | Area-weighted mean lat/lon |
| **Movement** | Centroid displacement vector per timestep |
| **Growth/Shrinkage** | Change in area per timestep |
| **Persistence** | Number of consecutive timesteps the event survives |
| **Severity Score** | Composite: f(intensity, area, persistence) |

#### G. Medium-Range Forecasting (Future Phase)
**Goal:** Predict anomaly maps 3–10 days ahead.
- Requires: Daily or 6-hourly ERA5 data + atmospheric driver variables
- Model candidates: ConvLSTM, U-Net, PredRNN
- Not implementable with current monthly data

### 2.3 Model Selection Rationale

| Model | Suitable For | Data Requirement | Current Viability |
|---|---|---|---|
| **Climatology baseline** | Anomaly computation | None | ✅ Already done |
| **Persistence baseline** | 1-step prediction | 3+ timesteps | ✅ Implement now |
| **Statistical threshold** | Extreme detection | 10+ samples per cell | ✅ Implement now |
| **XGBoost (tabular)** | Point forecast, feature importance | 60+ months ideally | ⚠️ Done; limited by data |
| **CNN (2D)** | Spatial pattern learning | 100+ monthly maps | ❌ Need extended ERA5 |
| **ConvLSTM** | Spatiotemporal dynamics | 200+ timesteps | ❌ Need extended ERA5 |
| **Graph Neural Network** | Complex teleconnections | Large graph dataset | ❌ Future research |

### 2.4 Atmospheric Variables for Medium-Range Forecasting (Do Not Download Yet)

For future medium-range forecasting (3–14 days ahead), the following atmospheric variables would add predictive skill:

| Variable | ERA5 Name | Physical Role |
|---|---|---|
| 10m U/V wind | `u10`, `v10` | Monsoon low-level jet positioning |
| Total column water vapour | `tcwv` | Moisture supply to precipitation systems |
| Sea surface temperature | `sst` | Ocean-atmosphere coupling, ENSO/IOD |
| CAPE | `cape` | Convective instability |
| 500 hPa geopotential | `z500` | Large-scale steering, block patterns |
| Outgoing longwave radiation | `mtnlwrf` | Convection activity proxy |
| 850 hPa vorticity | derived | Cyclonic circulation, low-pressure tracking |

---

## 3. DATA PIPELINE FOR NEXT PHASE

```
Current state:
  precipitation_anomaly.nc  [28 months, 125×121, monthly]

Required for Stage 3 (anomaly detection + tracking):
  No new data needed — use existing anomaly maps
  
Required for Stage 4 (CNN/XGBoost with real skill):
  Extend ERA5 to 1990-2024 (or at minimum 2010-2024)
  → ~168 usable months → adequate for XGBoost
  → ~300+ months → adequate for CNN

Required for Stage 6 (medium-range):
  ERA5 daily fields: tp, u10, v10, tcwv, z500
  Duration: 2010-2024
  Domain: India + surrounding ocean
  Estimated download size: ~50-100 GB
```

---

## 4. SCIENTIFIC CONSTRAINTS AND HONEST LIMITATIONS

1. **Short record (28 months):** The single monsoon cycle in training data (2021) is the fundamental bottleneck. No feature engineering or model architecture can compensate for this.

2. **Climatology robustness:** The 12-month climatological baseline is computed from only 2 full years. Some months have just 2 samples. This baseline is not robust by WMO standards (30-year normal is standard).

3. **No atmospheric dynamics:** Without wind, moisture, or ocean data, precipitation anomaly prediction is essentially extrapolation from precipitation history alone. This is physically underconstrained.

4. **Monthly resolution limits forecast horizon:** Monthly aggregation eliminates sub-monthly weather systems (cyclones, mesoscale convective systems). The model can only predict "was this month anomalous?", not "when did the extreme event occur or how long did it last?"

5. **India-domain only:** Teleconnections (ENSO, IOD, MJO) originate outside the India domain. SST anomalies in the Pacific and Indian Ocean need to be incorporated as boundary conditions.
