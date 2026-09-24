from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Coordinate(BaseModel):
    lat: float
    lon: float


class TimeSeriesPoint(BaseModel):
    date: str
    year: int
    month: int
    precipitation_mm: float
    climatology_mm: float
    anomaly_mm: float


class AnomalyResponse(BaseModel):
    query_lat: float
    query_lon: float
    grid_lat: float
    grid_lon: float
    total_months: int
    history: List[TimeSeriesPoint]


class ClimatologyPoint(BaseModel):
    month: int
    month_name: str
    climatology_mm: float


class ClimatologyResponse(BaseModel):
    query_lat: float
    query_lon: float
    grid_lat: float
    grid_lon: float
    annual_climatology_total_mm: float
    monthly_climatology: List[ClimatologyPoint]


class ForecastPoint(BaseModel):
    date: str
    month: int
    month_name: str
    climatology_mm: float
    forecast_anomaly_mm: float
    forecast_total_mm: float
    risk_category: str  # e.g., "Severe Deficit", "Moderate Deficit", "Normal", "Excess"


class ForecastResponse(BaseModel):
    query_lat: float
    query_lon: float
    grid_lat: float
    grid_lon: float
    forecast_steps: int
    forecasts: List[ForecastPoint]
    model_type: str
    scientific_limitation: str


class GridPoint(BaseModel):
    lat: float
    lon: float
    val: float


class GridResponse(BaseModel):
    date: str
    available_dates: List[str]
    min_val: float
    max_val: float
    mean_val: float
    resolution: float
    total_points: int
    points: List[GridPoint]


class LocationPreset(BaseModel):
    name: str
    state: str
    lat: float
    lon: float


class MetadataResponse(BaseModel):
    project_name: str
    dataset_source: str
    spatial_coverage: Dict[str, float]
    resolution_deg: float
    time_range: Dict[str, str]
    total_timesteps: int
    available_dates: List[str]
    model_status: str
    model_metrics: Dict[str, Any]
    locations: List[LocationPreset]
