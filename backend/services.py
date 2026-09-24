import math
import calendar
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import xarray as xr
import xgboost as xgb

from backend.models import (
    TimeSeriesPoint, AnomalyResponse,
    ClimatologyPoint, ClimatologyResponse,
    ForecastPoint, ForecastResponse,
    GridPoint, GridResponse,
    LocationPreset, MetadataResponse
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ANOMALY_NC = ROOT_DIR / "precipitation_anomaly.nc"
CLIMATOLOGY_NC = ROOT_DIR / "precipitation_climatology.nc"
MODEL_PATH = ROOT_DIR / "ml" / "models" / "xgb_baseline.json"
MODEL_CONFIG_PATH = ROOT_DIR / "ml" / "models" / "model_config.json"
SCALER_PATH = ROOT_DIR / "ml" / "models" / "scaler_params.json"
METRICS_TEST_PATH = ROOT_DIR / "ml" / "models" / "metrics_test.json"
METRICS_VAL_PATH = ROOT_DIR / "ml" / "models" / "metrics_validation.json"

# Major Indian cities for UI quick selection
PRESET_LOCATIONS = [
    LocationPreset(name="New Delhi", state="Delhi", lat=28.61, lon=77.21),
    LocationPreset(name="Mumbai", state="Maharashtra", lat=19.07, lon=72.87),
    LocationPreset(name="Bengaluru", state="Karnataka", lat=12.97, lon=77.59),
    LocationPreset(name="Chennai", state="Tamil Nadu", lat=13.08, lon=80.27),
    LocationPreset(name="Kolkata", state="West Bengal", lat=22.57, lon=88.36),
    LocationPreset(name="Hyderabad", state="Telangana", lat=17.38, lon=78.48),
    LocationPreset(name="Ahmedabad", state="Gujarat", lat=23.02, lon=72.57),
    LocationPreset(name="Guwahati", state="Assam", lat=26.14, lon=91.73),
    LocationPreset(name="Jaipur", state="Rajasthan", lat=26.91, lon=75.78),
    LocationPreset(name="Thiruvananthapuram", state="Kerala", lat=8.52, lon=76.93),
    LocationPreset(name="Shimla", state="Himachal Pradesh", lat=31.10, lon=77.17),
    LocationPreset(name="Patna", state="Bihar", lat=25.61, lon=85.14),
    LocationPreset(name="Bhopal", state="Madhya Pradesh", lat=23.25, lon=77.41),
    LocationPreset(name="Bhubaneswar", state="Odisha", lat=20.29, lon=85.82),
    LocationPreset(name="Srinagar", state="Jammu & Kashmir", lat=34.08, lon=74.79),
]


class DataService:
    def __init__(self):
        self.ds_anom = None
        self.ds_clim = None
        self.model = None
        self.model_config = {}
        self.scaler_params = {}
        self.metrics_test = {}
        self.metrics_val = {}
        self.available_dates = []
        self._load_all()

    def _load_all(self):
        # 1. Load NetCDF datasets
        if ANOMALY_NC.exists():
            self.ds_anom = xr.open_dataset(ANOMALY_NC)
            dates = [str(t)[:10] for t in self.ds_anom["valid_time"].values]
            self.available_dates = dates
        else:
            raise FileNotFoundError(f"Missing {ANOMALY_NC}")

        if CLIMATOLOGY_NC.exists():
            self.ds_clim = xr.open_dataset(CLIMATOLOGY_NC)
        else:
            raise FileNotFoundError(f"Missing {CLIMATOLOGY_NC}")

        # 2. Load ML model via Booster
        if MODEL_PATH.exists():
            self.model = xgb.Booster()
            self.model.load_model(str(MODEL_PATH))

        if MODEL_CONFIG_PATH.exists():
            with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)

        if SCALER_PATH.exists():
            with open(SCALER_PATH, "r", encoding="utf-8") as f:
                self.scaler_params = json.load(f)

        if METRICS_TEST_PATH.exists():
            with open(METRICS_TEST_PATH, "r", encoding="utf-8") as f:
                self.metrics_test = json.load(f)

        if METRICS_VAL_PATH.exists():
            with open(METRICS_VAL_PATH, "r", encoding="utf-8") as f:
                self.metrics_val = json.load(f)

    def get_metadata(self) -> MetadataResponse:
        lats = self.ds_anom["latitude"].values
        lons = self.ds_anom["longitude"].values
        return MetadataResponse(
            project_name="ERA5 Precipitation Analysis & Anomaly Forecasting",
            dataset_source="ERA5 Reanalysis (ECMWF)",
            spatial_coverage={
                "lat_min": float(lats.min()),
                "lat_max": float(lats.max()),
                "lon_min": float(lons.min()),
                "lon_max": float(lons.max())
            },
            resolution_deg=0.25,
            time_range={
                "start": self.available_dates[0],
                "end": self.available_dates[-1]
            },
            total_timesteps=len(self.available_dates),
            available_dates=self.available_dates,
            model_status="Trained (XGBoost Baseline)",
            model_metrics={
                "validation": self.metrics_val,
                "test": self.metrics_test
            },
            locations=PRESET_LOCATIONS
        )

    def get_anomaly_series(self, lat: float, lon: float) -> AnomalyResponse:
        """Extract historical monthly precipitation & anomaly series for nearest cell."""
        pt_anom = self.ds_anom["tp"].sel(latitude=lat, longitude=lon, method="nearest")
        pt_clim = self.ds_clim["tp"].sel(latitude=lat, longitude=lon, method="nearest")

        actual_lat = float(pt_anom.latitude)
        actual_lon = float(pt_anom.longitude)

        # 12-month climatology lookup
        clim_vals = {int(m): float(pt_clim.sel(month=m).values) for m in pt_clim["month"].values}

        points: List[TimeSeriesPoint] = []
        for t in pt_anom["valid_time"].values:
            date_str = str(t)[:10]
            year = int(date_str[:4])
            month = int(date_str[5:7])

            anom_val = float(pt_anom.sel(valid_time=t).values)
            clim_val = clim_vals.get(month, 0.0)
            precip_val = max(0.0, clim_val + anom_val)

            points.append(
                TimeSeriesPoint(
                    date=date_str,
                    year=year,
                    month=month,
                    precipitation_mm=round(precip_val, 2),
                    climatology_mm=round(clim_val, 2),
                    anomaly_mm=round(anom_val, 2)
                )
            )

        return AnomalyResponse(
            query_lat=lat,
            query_lon=lon,
            grid_lat=actual_lat,
            grid_lon=actual_lon,
            total_months=len(points),
            history=points
        )

    def get_climatology(self, lat: float, lon: float) -> ClimatologyResponse:
        """Extract 12-month seasonal cycle for nearest grid cell."""
        pt_clim = self.ds_clim["tp"].sel(latitude=lat, longitude=lon, method="nearest")
        actual_lat = float(pt_clim.latitude)
        actual_lon = float(pt_clim.longitude)

        points: List[ClimatologyPoint] = []
        total_clim = 0.0
        for m in sorted(int(x) for x in pt_clim["month"].values):
            val = float(pt_clim.sel(month=m).values)
            total_clim += val
            points.append(
                ClimatologyPoint(
                    month=m,
                    month_name=calendar.month_name[m],
                    climatology_mm=round(val, 2)
                )
            )

        return ClimatologyResponse(
            query_lat=lat,
            query_lon=lon,
            grid_lat=actual_lat,
            grid_lon=actual_lon,
            annual_climatology_total_mm=round(total_clim, 2),
            monthly_climatology=points
        )

    def generate_forecast(self, lat: float, lon: float, steps: int = 3) -> ForecastResponse:
        """Generate recursive multi-month forecast using trained XGBoost model."""
        steps = max(1, min(6, steps))
        hist_series = self.get_anomaly_series(lat, lon)
        clim_resp = self.get_climatology(lat, lon)
        clim_dict = {p.month: p.climatology_mm for p in clim_resp.monthly_climatology}

        # Need history for lags:
        # We need recent anomalies and precipitations
        recent_anoms = [p.anomaly_mm for p in hist_series.history]
        recent_precips = [p.precipitation_mm for p in hist_series.history]

        last_p = hist_series.history[-1]
        cur_year = last_p.year
        cur_month = last_p.month
        months_since_start = 28  # dataset is 28 months (0..27)

        forecasts: List[ForecastPoint] = []

        # Scaler lookup helper
        def scale(val: float, name: str) -> float:
            p = self.scaler_params.get(name, {"mean": 0.0, "std": 1.0})
            std = p["std"] if p["std"] != 0 else 1.0
            return (val - p["mean"]) / std

        # Spatial normalization stats from train set
        lat_norm_val = scale(hist_series.grid_lat, "lat")
        lon_norm_val = scale(hist_series.grid_lon, "lon")

        for step_i in range(steps):
            cur_month += 1
            if cur_month > 12:
                cur_month = 1
                cur_year += 1
            months_since_start += 1

            m_sin = math.sin(2 * math.pi * cur_month / 12)
            m_cos = math.cos(2 * math.pi * cur_month / 12)
            c_val = clim_dict.get(cur_month, 0.0)

            # Lag features from recent history
            anom_lag1 = recent_anoms[-1]
            anom_lag2 = recent_anoms[-2]
            anom_lag3 = recent_anoms[-3]
            prec_lag1 = recent_precips[-1]
            prec_lag2 = recent_precips[-2]

            roll_3m = float(np.mean(recent_precips[-3:]))
            roll_std_3m = float(np.std(recent_precips[-3:], ddof=0))
            roll_6m = float(np.mean(recent_precips[-6:]))

            # Assemble 18 features in exact order specified in model_config
            feat_dict = {
                "year": cur_year,
                "month": cur_month,
                "month_sin": m_sin,
                "month_cos": m_cos,
                "months_since_start": months_since_start,
                "lat": hist_series.grid_lat,
                "lon": hist_series.grid_lon,
                "lat_norm": lat_norm_val,
                "lon_norm": lon_norm_val,
                "climatology_mm": c_val,
                "anomaly_lag1": anom_lag1,
                "anomaly_lag2": anom_lag2,
                "anomaly_lag3": anom_lag3,
                "precip_lag1": prec_lag1,
                "precip_lag2": prec_lag2,
                "rolling_mean_3m": roll_3m,
                "rolling_std_3m": roll_std_3m,
                "rolling_mean_6m": roll_6m,
            }

            scaled_vector = []
            for col in self.model_config.get("feature_display", []):
                scaled_vector.append(scale(feat_dict[col], col))

            # Booster predict
            dmat = xgb.DMatrix(np.array([scaled_vector], dtype=np.float32))
            pred_anom = float(self.model.predict(dmat)[0])
            pred_total = max(0.0, c_val + pred_anom)

            # Determine risk category
            if pred_anom > 25.0:
                cat = "Excess Rainfall"
            elif pred_anom >= -25.0:
                cat = "Normal"
            elif pred_anom >= -75.0:
                cat = "Moderate Deficit"
            else:
                cat = "Severe Deficit"

            # Create date label (e.g. 2022-10-31)
            last_day = calendar.monthrange(cur_year, cur_month)[1]
            date_str = f"{cur_year}-{cur_month:02d}-{last_day:02d}"

            forecasts.append(
                ForecastPoint(
                    date=date_str,
                    month=cur_month,
                    month_name=calendar.month_name[cur_month],
                    climatology_mm=round(c_val, 2),
                    forecast_anomaly_mm=round(pred_anom, 2),
                    forecast_total_mm=round(pred_total, 2),
                    risk_category=cat
                )
            )

            # Update history for autoregressive next step
            recent_anoms.append(pred_anom)
            recent_precips.append(pred_total)

        return ForecastResponse(
            query_lat=lat,
            query_lon=lon,
            grid_lat=hist_series.grid_lat,
            grid_lon=hist_series.grid_lon,
            forecast_steps=steps,
            forecasts=forecasts,
            model_type="XGBoost Baseline Regressor",
            scientific_limitation=(
                "Model trained on 18 months of ERA5 data. Predictions tend toward the mean "
                "due to sample size constraints; treat as an exploratory demonstration baseline."
            )
        )

    def get_spatial_grid(self, date: str = None, step: int = 2) -> GridResponse:
        """Return 2D anomaly grid for map display. step=2 downsamples to ~3.8k points for fast transmission."""
        if not date or date not in self.available_dates:
            date = self.available_dates[-1]

        step = max(1, min(5, step))
        # Select date
        slice_2d = self.ds_anom["tp"].sel(valid_time=date)

        # Downsample slice if step > 1
        lats = slice_2d["latitude"].values[::step]
        lons = slice_2d["longitude"].values[::step]
        vals = slice_2d.values[::step, ::step]

        points: List[GridPoint] = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                v = float(vals[i, j])
                if not math.isnan(v):
                    points.append(
                        GridPoint(
                            lat=round(float(lat), 3),
                            lon=round(float(lon), 3),
                            val=round(v, 2)
                        )
                    )

        val_arr = np.array([p.val for p in points])
        return GridResponse(
            date=date,
            available_dates=self.available_dates,
            min_val=round(float(val_arr.min()), 2),
            max_val=round(float(val_arr.max()), 2),
            mean_val=round(float(val_arr.mean()), 2),
            resolution=0.25 * step,
            total_points=len(points),
            points=points
        )


# Global singleton instance
service = DataService()
