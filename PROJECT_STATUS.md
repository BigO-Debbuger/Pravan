# PROJECT_STATUS.md - Project Memory / Handoff Document

> **RULE:** Read this file FIRST at the start of every session.
> Update this file AFTER every meaningful development step.
> This is the single source of truth for project progress.
> Any AI continuing this project MUST read this file first, work only on the next unchecked task, and update this file before ending the session.

---

## 1. PROJECT OVERVIEW

| Field | Details |
|---|---|
| **Project Name** | VarshaVani: Smart India Hackathon (SIH) - Precipitation Analysis & Forecasting System |
| **Problem Being Solved** | India receives highly variable monsoon rainfall across diverse agro-climatic zones. There is a vital need for data-driven precipitation anomaly detection and predictive forecasting to support water resource management, agricultural planning, and disaster preparedness. |
| **Main Objective** | Build an end-to-end operational prototype that processes ERA5 reanalysis precipitation data for India, computes climatological baselines and anomalies, and trains robust spatiotemporal machine learning models to forecast future precipitation anomalies. |
| **Target Users** | Agricultural planners, meteorologists, disaster management officials, and water resource managers in India. |

---

## 2. CURRENT ARCHITECTURE

### ML Stack
- **Data Engine:** xarray + netCDF4 + numpy + pandas
- **Baseline Model:** XGBoost Regressor (`ml/train_model.py`, `ml/models/xgb_baseline.json`)
- **Input Features:** 18 spatiotemporal features (lags, rolling stats, cyclic harmonics, spatial coords)
- **Target:** `anomaly_mm` (continuous monthly precipitation anomaly in mm)
- **Split:** Monsoon-aware chronological split (Train: Dec 2020 - May 2022; Val: Jun - Jul 2022; Test: Aug - Sep 2022 held-out)
- **Model Inference Runtime:** XGBoost `xgb.Booster` (Python 3.13 compatibility)

### Data Pipeline
```
ERA5 Raw NetCDF (hourly, m/hr)
  -> Convert to mm
  -> Resample to daily totals
  -> Resample to monthly totals
  -> Compute 12-month climatology (mean per calendar month)
  -> Compute monthly anomaly = monthly_total - climatology
  -> Save: precipitation_climatology.nc, precipitation_anomaly.nc
  -> Feature engineering (ml/features.py)
  -> 18-feature matrix per (month, lat, lon) cell
  -> Monsoon-aware chronological split (Train: 18mo / Val: 2mo / Test: 2mo)
  -> StandardScaler fit on train only -> ml/models/scaler_params.json
  -> Save: ml/processed/features_{train,val,test}.parquet
  -> Train XGBoost baseline -> ml/models/xgb_baseline.json
```

### Important Directories
```
SIH Second statement/
+-- 4572d4b99342c81e5c37ac789460b190/
|   +-- data_stream-oper_stepType-accum.nc        <- Raw ERA5 (169 MB)
+-- ml/
|   +-- preprocessing.py                          <- Data loader module
|   +-- features.py                               <- Feature engineering pipeline
|   +-- train_model.py                            <- XGBoost baseline training script
|   +-- processed/
|   |   +-- features_train.parquet (272,250 rows, 18 months, Dec 2020 - May 2022)
|   |   +-- features_val.parquet   (30,250 rows, 2 months, Jun - Jul 2022)
|   |   +-- features_test.parquet  (30,250 rows, 2 months, Aug - Sep 2022)
|   |   +-- feature_names.txt
|   |   +-- split_summary.txt
|   +-- models/
|       +-- xgb_baseline.json                     <- Trained XGBoost baseline model
|       +-- model_config.json                     <- Hyperparameters and config
|       +-- scaler_params.json                    <- Scaler mean/std per feature
|       +-- metrics_validation.json / .txt        <- Validation evaluation metrics
|       +-- metrics_test.json / .txt              <- Held-out test evaluation metrics
|       +-- feature_importance.csv / .png         <- Top features by gain
|       +-- validation_predictions.png            <- Actual vs predicted scatter & residuals
+-- inspect_era5.py
+-- calculate_climatology.py
+-- plot_era5.py
+-- plot_anomaly.py
+-- precipitation_climatology.nc
+-- precipitation_anomaly.nc
+-- era5_first_map.png
+-- era5_anomaly_map.png
+-- PROJECT_STATUS.md                             <- THIS FILE
```

*(Note: There are frontend/backend directories on disk from prior autonomous work, but they are excluded here as we are focusing solely on the current verified ML state per instructions.)*

---

## 3. DATASET INFORMATION

| Field | Details |
|---|---|
| **Dataset Source** | ERA5 Reanalysis — ECMWF via CDS |
| **Variable** | `tp` — Total Precipitation (accumulated, surface level) |
| **Raw File Path** | `4572d4b99342c81e5c37ac789460b190/data_stream-oper_stepType-accum.nc` |
| **Raw File Size** | ~169 MB (176,745,628 bytes) |
| **Format** | NetCDF (CF-1.7) |
| **Time Range** | 2020-06-01 to 2022-09-30 (hourly, 8,784 timesteps) |
| **Spatial Coverage** | Lat 6°N - 37°N, Lon 68°E - 98°E (India domain) |
| **Spatial Resolution** | 0.25° x 0.25° (~28 km grid spacing) |
| **Grid Dimensions** | 125 lat x 121 lon = 15,125 grid cells per timestep |
| **Raw Units** | Metres (m) |
| **Processed Units** | Millimetres (mm) |

### ML Feature Matrix (18 features)

| Feature | Type | Description |
|---|---|---|
| year | temporal | Calendar year (trend proxy) |
| month | temporal | Calendar month 1-12 (seasonality) |
| month_sin | temporal | sin(2pi*month/12) — cyclic encoding |
| month_cos | temporal | cos(2pi*month/12) — cyclic encoding |
| months_since_start | temporal | Integer index 0,1,2,... (linear trend) |
| lat | spatial | Latitude of grid cell |
| lon | spatial | Longitude of grid cell |
| lat_norm | spatial | Normalised latitude (z-score, train stats) |
| lon_norm | spatial | Normalised longitude (z-score, train stats) |
| climatology_mm | climatology | Long-term mean for that calendar month at that cell (mm) |
| anomaly_lag1 | lag | Anomaly 1 month prior (t-1), mm |
| anomaly_lag2 | lag | Anomaly 2 months prior (t-2), mm |
| anomaly_lag3 | lag | Anomaly 3 months prior (t-3), mm |
| precip_lag1 | lag | Total rainfall 1 month prior, mm |
| precip_lag2 | lag | Total rainfall 2 months prior, mm |
| rolling_mean_3m | rolling | Mean of last 3 months total rainfall (t-3 to t-1), mm |
| rolling_std_3m | rolling | Std of last 3 months total rainfall, mm |
| rolling_mean_6m | rolling | Mean of last 6 months total rainfall (t-6 to t-1), mm |

**Target:** `anomaly_mm` — continuous precipitation anomaly (mm) at month t

---

## 4. COMPLETED WORK

```
[x] ERA5 dataset downloaded (NetCDF, hourly, 2020-06 to 2022-09, India domain)
[x] NetCDF successfully opened with xarray
[x] Dataset inspected (dimensions, coordinates, variables, attributes confirmed)
[x] Hourly precipitation converted to mm and resampled to daily totals
[x] Monthly totals computed from daily data
[x] Monthly climatology calculated (12-month mean baseline)
[x] Precipitation anomaly calculated (monthly total minus climatology)
[x] climatology saved -> precipitation_climatology.nc
[x] anomaly saved -> precipitation_anomaly.nc
[x] Initial precipitation map generated (era5_first_map.png)
[x] Anomaly map generated (era5_anomaly_map.png)
[x] Prediction problem designed (regression on continuous anomaly mm)
[x] ml/preprocessing.py created and tested (loads climatology + anomaly files)
[x] ml/features.py created and tested (18-feature pipeline, no data leakage)
[x] Feature matrix saved: ml/processed/features_{train,val,test}.parquet
[x] NaN lag rows dropped correctly (first 6 months, 90,750 rows removed)
[x] Monsoon-aware chronological split: train 18mo (272,250 rows) / val 2mo (30,250 rows) / test 2mo (30,250 rows)
[x] StandardScaler fit on TRAIN only, applied to val+test
[x] Feature names and split summary saved to ml/processed/
[x] Baseline XGBoost model trained with early stopping (ml/train_model.py)
[x] Model evaluation and scientific diagnostics saved (ml/models/)
[x] Fix plot_anomaly.py (loads precipitation_anomaly.nc directly in <2s)
```

---

## 5. CURRENT FILES

| File | Purpose |
|---|---|
| `4572d4b99342c81e5c37ac789460b190/data_stream-oper_stepType-accum.nc` | Raw ERA5 hourly tp, India, Jun 2020-Sep 2022 |
| `inspect_era5.py` | Prints dataset structure |
| `calculate_climatology.py` | Preprocessing: mm conversion -> daily -> monthly -> climatology -> anomaly |
| `plot_era5.py` | First hourly timestep map |
| `plot_anomaly.py` | Anomaly map (last month) — optimized direct NetCDF loader |
| `precipitation_climatology.nc` | Derived: 12-month mean baseline (12 x 125 x 121) |
| `precipitation_anomaly.nc` | Derived: monthly anomaly (28 x 125 x 121) |
| `era5_first_map.png` | Map image: first timestep precipitation |
| `era5_anomaly_map.png` | Map image: last month anomaly (Sep 2022) |
| `ml/preprocessing.py` | Data loader: loads climatology + anomaly NetCDF files, handles NaN from GRIB artefacts |
| `ml/features.py` | Feature engineering pipeline: builds 18-feature matrix, lag/rolling features, cyclic encoding, chronological split, StandardScaler (train-only), saves parquet files |
| `ml/train_model.py` | Baseline XGBoost regressor training pipeline, early stopping, evaluation, and visualization |
| `ml/processed/features_train.parquet` | Train set: 272,250 rows, 18 months (Dec 2020 - May 2022) |
| `ml/processed/features_val.parquet` | Val set: 30,250 rows, 2 months (Jun - Jul 2022, early monsoon) |
| `ml/processed/features_test.parquet` | Test set: 30,250 rows, 2 months (Aug - Sep 2022, late monsoon held-out) |
| `ml/processed/feature_names.txt` | List of 18 raw + 18 scaled feature column names |
| `ml/processed/split_summary.txt` | Split details + scaler params per feature |
| `ml/models/xgb_baseline.json` | Trained XGBoost baseline model artifact |
| `ml/models/model_config.json` | Hyperparameters, feature list, and metadata |
| `ml/models/scaler_params.json` | Scaler mean and std per feature for backend inference |
| `ml/models/metrics_validation.json` / `.txt` | Validation metrics (Jun-Jul 2022) |
| `ml/models/metrics_test.json` / `.txt` | Held-out test metrics (Aug-Sep 2022) |
| `ml/models/feature_importance.csv` / `.png` | Feature importances ranked by gain |
| `ml/models/validation_predictions.png` | Scatter plot & residuals visualization |
| `PROJECT_STATUS.md` | THIS FILE |

---

## 6. CURRENT RESULTS

> Only verified, real results. No invented metrics.

### Dataset
- 8,784 hourly timesteps x 125 lat x 121 lon confirmed
- Time range: 2020-06-01 to 2022-09-30

### Feature Matrix (verified, pipeline ran successfully with exit code 0)
- Total rows (all splits combined): 332,750
- First 6 months dropped (90,750 rows) — incomplete 6-month rolling window
- No missing values in any feature or target column

### Split Sizes (Monsoon-Aware Chronological Split)
| Split | Months | Date Range | Rows | Target Regime |
|---|---|---|---|---|
| Train | 18 | 2020-12-31 to 2022-05-31 | 272,250 | Includes 2021 monsoon (Jun-Sep) |
| Val   | 2  | 2022-06-30 to 2022-07-31 | 30,250  | 2022 early monsoon |
| Test  | 2  | 2022-08-31 to 2022-09-30 | 30,250  | 2022 late monsoon (held-out) |

### Target Statistics (train split)
| Stat | Value |
|---|---|
| min | -755.68 mm |
| max | +736.68 mm |
| mean | -2.32 mm |
| std | 40.03 mm |
| median | 0.00 mm |

### ML Baseline Model Performance (XGBoost Regressor)

#### Validation Set (Jun - Jul 2022, 30,250 samples)
| Metric | Value |
|---|---|
| MAE | 59.0602 mm |
| RMSE | 89.5691 mm |
| R² | -0.0043 |
| Actual Mean ± Std | -6.40 mm ± 89.38 mm |
| Predicted Mean ± Std | -2.69 mm ± 1.24 mm |
| Pred Std / Actual Std | 0.0138 (predictions collapse near regional mean) |

#### Test Set (Aug - Sep 2022, 30,250 samples, held-out)
| Metric | Value |
|---|---|
| MAE | 48.4801 mm |
| RMSE | 70.6850 mm |
| R² | -0.0169 |
| Actual Mean ± Std | -2.57 mm ± 70.10 mm |
| Predicted Mean ± Std | -3.14 mm ± 3.14 mm |
| Pred Std / Actual Std | 0.0448 |

#### Feature Importance (Top Features by Gain)
1. `month` (0.428)
2. `month_sin` (0.244)
3. `anomaly_lag3` (0.136)
4. `lat_norm` (0.053)
5. `anomaly_lag2` (0.047)

#### Scientific Diagnosis & Limitations
- **Mean Collapse:** The baseline model exhibits mean-collapse behavior due to the very small training sample (only approximately one historical monsoon season). 
- **Status:** The XGBoost baseline is useful as a benchmark but is NOT sufficient as the final spatio-temporal forecasting model.

---

## 7. CURRENT PROBLEMS / BUGS

| # | Issue | Status |
|---|---|---|
| 1 | Dataset covers only ~28 months. Climatological baseline may not be robust for all 12 months (some months have 1-2 samples only). | Known limitation |
| 2 | Baseline XGBoost model trained; reveals expected mean collapse due to 18-month training limit. | Handled & documented |
| 3 | NaN values exist in daily data (7,350,750 NaN from GRIB accumulation artefacts). | Handled correctly in preprocessing.py |

---

## 8. NEXT STEPS (Priority Order)

1. [ ] **Design and Build Stronger Spatio-Temporal Model** — The current XGBoost baseline lacks physical forecasting depth. The next phase must focus on a robust spatiotemporal modeling approach to replace the baseline.

---

## 9. IMPORTANT TECHNICAL DECISIONS

| Decision | Rationale |
|---|---|
| **Regression (continuous), not classification** | With only 28 monthly timesteps, regression preserves more information. |
| **6-month rolling window as max lag** | Drives the decision to drop the first 6 months. Larger windows would leave too few training samples. |
| **Chronological split (no shuffle)** | Time-series data — random shuffle would cause severe data leakage. |
| **StandardScaler fit on TRAIN only** | Fitting on the full dataset before splitting would leak val/test distribution into training. |
| **XGBoost Booster runtime for Python 3.13** | Using `xgb.Booster().load_model()` bypasses scikit-learn Python 3.13 tag deprecation warnings and loads in <5ms. |

---

## 10. RUN COMMANDS

### Install Dependencies
```bash
pip install xarray numpy matplotlib netcdf4 scipy pyarrow pandas scikit-learn xgboost
```

### Run Feature Engineering & Training
```bash
# Feature generation
python ml/features.py

# Model training & evaluation
python ml/train_model.py
```

### Generate Visualizations
```bash
python plot_era5.py       # -> era5_first_map.png
python plot_anomaly.py    # -> era5_anomaly_map.png (<2s)
```

---

## 11. LAST UPDATED

| Field | Value |
|---|---|
| **Date/Time** | 2026-09-24 |
| **Updated By** | Antigravity AI |
| **Latest Work** | Rewrote PROJECT_STATUS.md to reflect the current, verified ML state. Created GitHub checkpoint. |
| **Current Phase** | Preparing to design and implement a robust spatio-temporal forecasting model. |

---

## 12. GitHub Checkpoint

- **Checkpoint Commit:** `aa8d22a` (Add ERA5 preprocessing and XGBoost baseline)
- **Date:** 2026-09-24
- **Committed:** ML preprocessing scripts, feature engineering pipeline, XGBoost baseline code, Parquet splits, model configs, and existing frontend/backend code.
- **Intentionally Excluded:** `*.nc` (raw ERA5 ~169MB dataset, generated climatology/anomaly NetCDFs) and `*.png` visualizations.
- **Status:** Raw dataset remains local. The repository is ready for the next ML architecture phase.
