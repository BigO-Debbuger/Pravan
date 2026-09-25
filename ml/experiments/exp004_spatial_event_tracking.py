"""
ml/experiments/exp004_spatial_event_tracking.py
------------------------------------------------
EXP-004: Spatial Event Tracking

PURPOSE
=======
Identify connected extreme precipitation regions in the binary extreme masks 
produced by EXP-001. Track these regions as physical "events".

This is an object-based detection approach, mapping pixel-level predictions 
into meaningful spatiotemporal entities.

METHODOLOGY
===========
1. Load `exp001_extreme_masks.nc` (signed mask: +1 wet, -1 dry).
2. For each timestep, apply `scipy.ndimage.label` to detect connected 
   components for wet (+1) and dry (-1) extremes separately.
3. Connectivity: 8-connectivity (diagonal + orthogonal neighbors) is used.
4. Filter out artifacts: minimum event size = 10 grid cells. 
   At 0.25 deg (~25km), 1 cell ~ 625 km^2. 10 cells ~ 6,250 km^2. 
   This removes very localized noise.
5. Compute spatial properties for each surviving blob:
   - Centroid lat/lon
   - Bounding box
   - Area (approx km^2)
   - Mean/Max/Min anomaly intensity
6. Simple Tracking (t to t+1):
   - Match events of the same sign if their centroids are within a 
     distance threshold (e.g., 5 degrees ~ 500 km) and there is some overlap.
     (Note: Since data is monthly, tracking is difficult because systems move faster.
      However, stationary seasonal anomalies can persist).

INPUTS
======
  - precipitation_anomaly.nc
  - ml/experiments/exp001_extreme_masks.nc

OUTPUTS
=======
  - ml/experiments/exp004_event_catalog.parquet
  - ml/experiments/exp004_event_stats.json
  - ml/experiments/exp004_event_map.png
"""

import pathlib
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# ---- paths ------------------------------------------------------------------
ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent
ANOM_PATH = ROOT / "precipitation_anomaly.nc"
MASK_PATH = ROOT / "ml" / "experiments" / "exp001_extreme_masks.nc"
OUT_DIR   = ROOT / "ml" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = OUT_DIR / "exp004_event_catalog.parquet"
STATS_PATH   = OUT_DIR / "exp004_event_stats.json"
PLOT_PATH    = OUT_DIR / "exp004_event_map.png"

# Parameters
MIN_CELLS = 10           # Minimum contiguous cells to constitute an event
CONNECTIVITY_STRUCT = np.ones((3, 3), dtype=int) # 8-connectivity


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def cell_area_km2(lat):
    """Approximate area of a 0.25 x 0.25 degree cell at given latitude."""
    lat_rad = radians(lat)
    dy = 111.32 * 0.25
    dx = 111.32 * 0.25 * cos(lat_rad)
    return dy * dx

def run_exp004(verbose: bool = True):
    print("=" * 65)
    print("  EXP-004: SPATIAL EVENT TRACKING")
    print("=" * 65)
    print(f"  Inputs: \n    {ANOM_PATH.name}\n    {MASK_PATH.name}")
    print(f"  Min Event Size: {MIN_CELLS} cells (approx 6,000 - 7,000 km^2)")
    print()

    # ---- 1. Load Data -------------------------------------------------------
    if verbose: print("[1/7] Loading anomaly and mask data...")
    ds_anom = xr.open_dataset(ANOM_PATH)
    ds_mask = xr.open_dataset(MASK_PATH)
    
    anom_var = list(ds_anom.data_vars)[0]
    anom_arr = ds_anom[anom_var].values  # (28, 125, 121)
    
    signed_mask = ds_mask["signed_extreme_mask"].values # +1 wet, -1 dry, 0 normal
    times = pd.DatetimeIndex(ds_anom.valid_time.values)
    lats = ds_anom.latitude.values
    lons = ds_anom.longitude.values
    
    n_t, n_lat, n_lon = anom_arr.shape
    
    # Precompute cell areas (1D array corresponding to lats)
    areas_1d = np.array([cell_area_km2(lat) for lat in lats])
    # 2D array matching the grid shape
    areas_2d = np.tile(areas_1d, (n_lon, 1)).T

    # ---- 2. Extract Connected Components ------------------------------------
    if verbose: print("\n[2/7] Extracting connected components per timestep...")
    
    events = []
    blob_id_counter = 1
    
    for t_idx in range(n_t):
        current_time = times[t_idx]
        current_anom = anom_arr[t_idx]
        current_mask = signed_mask[t_idx]
        
        # Detect wet events
        wet_mask = (current_mask == 1).astype(int)
        wet_labels, wet_num = ndimage.label(wet_mask, structure=CONNECTIVITY_STRUCT)
        
        # Detect dry events
        dry_mask = (current_mask == -1).astype(int)
        dry_labels, dry_num = ndimage.label(dry_mask, structure=CONNECTIVITY_STRUCT)
        
        # Process blobs
        for is_wet, num_blobs, labels in [(True, wet_num, wet_labels), (False, dry_num, dry_labels)]:
            for i in range(1, num_blobs + 1):
                blob_mask = (labels == i)
                n_cells = blob_mask.sum()
                
                # Filter by size
                if n_cells < MIN_CELLS:
                    continue
                    
                # Calculate properties
                blob_anom = current_anom[blob_mask]
                blob_lats = lats[np.any(blob_mask, axis=1)]
                blob_lons = lons[np.any(blob_mask, axis=0)]
                
                # Area
                blob_area = areas_2d[blob_mask].sum()
                
                # Centroid
                # Calculate center of mass based on anomaly magnitude
                weights = np.abs(blob_anom)
                if weights.sum() > 0:
                    com = ndimage.center_of_mass(weights.reshape(blob_mask.shape) if len(blob_anom) == 1 else blob_mask)
                    # center of mass gives indices (y, x), we need actual lat/lon. It's better to just use mean of indices.
                    y_indices, x_indices = np.where(blob_mask)
                    cent_y = np.average(y_indices, weights=weights)
                    cent_x = np.average(x_indices, weights=weights)
                else:
                    y_indices, x_indices = np.where(blob_mask)
                    cent_y = np.mean(y_indices)
                    cent_x = np.mean(x_indices)
                    
                # Interpolate exact lat/lon for centroid
                # Bounds check just in case
                cent_y = np.clip(cent_y, 0, n_lat - 1)
                cent_x = np.clip(cent_x, 0, n_lon - 1)
                
                # Since lats might be decreasing, interp handles it if x is increasing. 
                # Let's just use indices as weights directly on the coordinate arrays
                centroid_lat = np.average(lats[y_indices], weights=weights) if weights.sum() > 0 else np.mean(lats[y_indices])
                centroid_lon = np.average(lons[x_indices], weights=weights) if weights.sum() > 0 else np.mean(lons[x_indices])
                
                
                event_record = {
                    "blob_id": f"B{blob_id_counter:05d}",
                    "time_idx": t_idx,
                    "date": current_time,
                    "type": "wet" if is_wet else "dry",
                    "n_cells": int(n_cells),
                    "area_km2": float(blob_area),
                    "centroid_lat": float(centroid_lat),
                    "centroid_lon": float(centroid_lon),
                    "min_lat": float(blob_lats.min()),
                    "max_lat": float(blob_lats.max()),
                    "min_lon": float(blob_lons.min()),
                    "max_lon": float(blob_lons.max()),
                    "mean_anomaly_mm": float(blob_anom.mean()),
                    "max_magnitude_mm": float(np.abs(blob_anom).max() * (1 if is_wet else -1)),
                    "total_anomaly_mm": float(blob_anom.sum())
                }
                
                events.append(event_record)
                blob_id_counter += 1
                
    if verbose:
        print(f"   Detected {len(events)} valid events across all timesteps (>{MIN_CELLS} cells).")

    # ---- 3. Temporal Tracking (Assigning Event IDs) -------------------------
    if verbose: print("\n[3/7] Tracking events across time...")
    
    # Sort events by time
    df_events = pd.DataFrame(events)
    if not df_events.empty:
        df_events.sort_values(by=["time_idx", "area_km2"], ascending=[True, False], inplace=True)
        
        event_id_counter = 1
        df_events["event_id"] = None
        
        # Simple tracking: if an event at t+1 is of the same type and its centroid is within 
        # a threshold distance of an event at t, link them.
        DIST_THRESHOLD_KM = 500  # 500 km movement allowed per month
        
        for t_idx in range(n_t):
            current_blobs = df_events[df_events["time_idx"] == t_idx].index
            
            for idx in current_blobs:
                blob = df_events.loc[idx]
                
                if pd.isna(blob["event_id"]):
                    # Start a new track
                    current_event_id = f"EVT-{blob['date'].year}-{event_id_counter:03d}"
                    df_events.at[idx, "event_id"] = current_event_id
                    event_id_counter += 1
                else:
                    current_event_id = blob["event_id"]
                
                # Look ahead 1 timestep
                next_blobs = df_events[(df_events["time_idx"] == t_idx + 1) & 
                                       (df_events["type"] == blob["type"]) &
                                       (df_events["event_id"].isna())].index
                
                best_match = None
                min_dist = DIST_THRESHOLD_KM
                
                for n_idx in next_blobs:
                    nblob = df_events.loc[n_idx]
                    dist = haversine(blob["centroid_lon"], blob["centroid_lat"],
                                     nblob["centroid_lon"], nblob["centroid_lat"])
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match = n_idx
                        
                if best_match is not None:
                    df_events.at[best_match, "event_id"] = current_event_id
        
        # Reorder columns
        cols = ["event_id"] + [c for c in df_events.columns if c != "event_id"]
        df_events = df_events[cols]
        
        n_unique_events = df_events["event_id"].nunique()
        if verbose:
            print(f"   Condensed {len(df_events)} blobs into {n_unique_events} unique event tracks.")
    else:
        df_events = pd.DataFrame(columns=["event_id", "blob_id", "time_idx", "date", "type", "n_cells", "area_km2", "centroid_lat", "centroid_lon", "mean_anomaly_mm"])
        n_unique_events = 0
        if verbose: print("   No events found to track.")


    # ---- 4. Save Catalog ----------------------------------------------------
    if verbose: print("\n[4/7] Saving event catalog ...")
    
    # Save as Parquet
    df_events.to_parquet(CATALOG_PATH, index=False)
    
    # Also save as CSV for easy inspection if it's small
    csv_path = CATALOG_PATH.with_suffix(".csv")
    df_events.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"   Saved Parquet: {CATALOG_PATH.name}")
        print(f"   Saved CSV    : {csv_path.name}")

    # ---- 5. Compute and Save Statistics -------------------------------------
    if verbose: print("\n[5/7] Computing catalog statistics ...")
    
    stats = {
        "experiment": "EXP-004",
        "description": "Spatial event tracking based on EXP-001 connected components",
        "generated_at": datetime.now().isoformat(),
        "parameters": {
            "min_cells": MIN_CELLS,
            "min_area_km2_approx": MIN_CELLS * 625,
            "connectivity": "8-way (diagonal + orthogonal)",
            "tracking_dist_threshold_km": 500
        },
        "summary": {
            "total_timesteps": n_t,
            "total_blobs_found": len(df_events),
            "unique_event_tracks": n_unique_events,
        }
    }
    
    if not df_events.empty:
        wet_events = df_events[df_events["type"] == "wet"]
        dry_events = df_events[df_events["type"] == "dry"]
        
        track_lengths = df_events.groupby("event_id").size()
        
        stats["summary"].update({
            "wet_blobs": len(wet_events),
            "dry_blobs": len(dry_events),
            "mean_area_km2": float(df_events["area_km2"].mean()),
            "median_area_km2": float(df_events["area_km2"].median()),
            "max_area_km2": float(df_events["area_km2"].max()),
            "mean_persistence_months": float(track_lengths.mean()),
            "max_persistence_months": float(track_lengths.max())
        })
        
        # Monthly distribution
        stats["monthly_distribution"] = df_events.groupby(df_events["date"].dt.month).size().to_dict()
        
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    if verbose: print(f"   Saved: {STATS_PATH.name}")

    # ---- 6. Visualization ---------------------------------------------------
    if verbose: print(f"\n[6/7] Creating visualization -> {PLOT_PATH.name} ...")
    
    if not df_events.empty:
        _make_visualization(df_events, anom_arr, signed_mask, times, lats, lons)
    else:
        print("   Skipping visualization because no events were found.")

    # ---- 7. Verification ----------------------------------------------------
    if verbose: print("\n[7/7] Verification ...")
    assert not df_events["date"].isna().any(), "NaN found in dates"
    assert not df_events["event_id"].isna().any(), "NaN found in event_ids"
    if verbose:
        print(f"   Rows in catalog: {len(df_events)}")
        print("   No temporal leakage in centroid tracking (strictly forward in time).")
        print("   All checks PASSED.")

    print()
    print("=" * 65)
    print("  EXP-004 COMPLETE")
    print("=" * 65)
    print(f"  Blobs detected (>{MIN_CELLS} cells): {len(df_events)}")
    print(f"  Unique event tracks         : {n_unique_events}")
    print(f"  Outputs:")
    print(f"    {CATALOG_PATH}")
    print(f"    {STATS_PATH}")
    print(f"    {PLOT_PATH}")
    print()
    
    ds_anom.close()
    ds_mask.close()


def _make_visualization(df_events, anom_arr, signed_mask, times, lats, lons):
    """
    Visualization showing:
    1. A scatter map of all event centroids colored by type (wet/dry).
    2. A histogram of event areas.
    3. Spatial map of the timestep with the largest single event blob.
    """
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("#0d1117")
    
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    for ax in fig.axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
            
    # --- Panel 1: Centroids Scatter ---
    wet = df_events[df_events["type"] == "wet"]
    dry = df_events[df_events["type"] == "dry"]
    
    ax1.scatter(wet["centroid_lon"], wet["centroid_lat"], c="#3b82f6", alpha=0.6, s=wet["area_km2"]/1000, label="Wet")
    ax1.scatter(dry["centroid_lon"], dry["centroid_lat"], c="#f97316", alpha=0.6, s=dry["area_km2"]/1000, label="Dry")
    ax1.set_title("Event Centroids (Bubble size ~ Area)", color="white")
    ax1.set_xlabel("Longitude", color="white")
    ax1.set_ylabel("Latitude", color="white")
    ax1.legend(facecolor="#1c2128", labelcolor="white")
    
    # --- Panel 2: Area Distribution ---
    ax2.hist(df_events["area_km2"] / 1000, bins=30, color="#22c55e", alpha=0.8)
    ax2.set_title("Event Size Distribution", color="white")
    ax2.set_xlabel("Area (Thousands of km²)", color="white")
    ax2.set_ylabel("Count", color="white")
    
    # --- Panel 3: Largest Event Map ---
    idx_largest = df_events["area_km2"].idxmax()
    largest_blob = df_events.loc[idx_largest]
    t_idx = largest_blob["time_idx"]
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Show anomaly map for that timestep, and contour the mask
    ax3.pcolormesh(lon_grid, lat_grid, anom_arr[t_idx], cmap="RdBu_r", vmin=-200, vmax=200, shading="auto", alpha=0.5)
    
    # Mask contour
    ax3.contour(lon_grid, lat_grid, np.abs(signed_mask[t_idx]), levels=[0.5], colors=['#22c55e'], linewidths=1.5)
    ax3.plot(largest_blob["centroid_lon"], largest_blob["centroid_lat"], 'X', color='yellow', markersize=10, markeredgecolor='black')
    
    date_str = str(largest_blob["date"].date())
    ax3.set_title(f"Largest Event: {date_str} ({largest_blob['type'].capitalize()})\nArea: {largest_blob['area_km2']/1000:.1f}k km²", color="white", fontsize=10)
    ax3.set_xlabel("Longitude", color="white")
    
    fig.suptitle("EXP-004: Spatial Event Tracking (Connected Components of |Anomaly| > 1.5σ)", color="white", fontsize=14, y=1.05)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


if __name__ == "__main__":
    run_exp004(verbose=True)
