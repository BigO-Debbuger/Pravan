import os
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    AnomalyResponse, ClimatologyResponse,
    ForecastResponse, GridResponse, MetadataResponse
)
from backend.services import service

app = FastAPI(
    title="India Precipitation Anomaly & Forecasting API",
    description="ERA5 Reanalysis Precipitation Climatology, Anomaly Detection & Machine Learning Forecasting System for India.",
    version="1.0.0"
)

# Enable CORS for local dev servers and frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", summary="Health check endpoint")
def health_check():
    return {
        "status": "online",
        "service": "India Precipitation Forecasting API",
        "dataset_timesteps": len(service.available_dates),
        "latest_date": service.available_dates[-1] if service.available_dates else None,
        "model_loaded": service.model is not None
    }


@app.get("/api/metadata", response_model=MetadataResponse, summary="Get dataset coverage, model metrics, and city presets")
def get_metadata():
    try:
        return service.get_metadata()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomaly", response_model=AnomalyResponse, summary="Get historical monthly precipitation and anomaly series for coordinates")
def get_anomaly(
    lat: float = Query(..., description="Latitude in decimal degrees (6.0 to 37.0)", ge=6.0, le=37.0),
    lon: float = Query(..., description="Longitude in decimal degrees (68.0 to 98.0)", ge=68.0, le=98.0)
):
    try:
        return service.get_anomaly_series(lat=lat, lon=lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query anomaly series: {str(e)}")


@app.get("/api/climatology", response_model=ClimatologyResponse, summary="Get 12-month baseline climatology for coordinates")
def get_climatology(
    lat: float = Query(..., description="Latitude in decimal degrees (6.0 to 37.0)", ge=6.0, le=37.0),
    lon: float = Query(..., description="Longitude in decimal degrees (68.0 to 98.0)", ge=68.0, le=98.0)
):
    try:
        return service.get_climatology(lat=lat, lon=lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query climatology: {str(e)}")


@app.get("/api/forecast", response_model=ForecastResponse, summary="Generate multi-month recursive forecast using ML baseline")
def get_forecast(
    lat: float = Query(..., description="Latitude in decimal degrees (6.0 to 37.0)", ge=6.0, le=37.0),
    lon: float = Query(..., description="Longitude in decimal degrees (68.0 to 98.0)", ge=68.0, le=98.0),
    months: int = Query(3, description="Number of months to forecast ahead (1 to 6)", ge=1, le=6)
):
    try:
        return service.generate_forecast(lat=lat, lon=lon, steps=months)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")


@app.get("/api/grid", response_model=GridResponse, summary="Get 2D spatial grid of anomalies for India map visualization")
def get_grid(
    date: Optional[str] = Query(None, description="Valid time string (YYYY-MM-DD), defaults to latest"),
    step: int = Query(2, description="Downsampling stride (1=full ~15k points, 2=fast ~3.8k points, 3=superfast ~1.7k points)", ge=1, le=5)
):
    try:
        return service.get_spatial_grid(date=date, step=step)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load spatial grid: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
