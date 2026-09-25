"""
download_era5_chunked.py
------------------------
Robust, resumable ERA5 precipitation downloader (one year per request).

USAGE
-----
  # Download all missing years (resumes automatically):
  python download_era5_chunked.py

  # Download specific year only:
  python download_era5_chunked.py --year 2015

  # Merge all completed yearly files into one:
  python download_era5_chunked.py --merge-only

  # Verify downloaded files without downloading:
  python download_era5_chunked.py --verify-only

CONFIGURATION
-------------
  Credentials are loaded from .env:
    CDS_API_URL=https://cds.climate.copernicus.eu/api
    CDS_API_KEY=<your_token>

  Domain / resolution match the existing ERA5 dataset:
    lat: 37N -> 6N, lon: 68E -> 98E, 0.25deg, hourly, tp

OUTPUT
------
  data/era5_extended/yearly/era5_tp_india_<YEAR>.nc  (one per year)
  era5_tp_india_2010_2024.nc                          (final merged file)
"""

import os
import sys
import json
import time
import pathlib
import argparse
import traceback
from datetime import datetime

from dotenv import load_dotenv

# ---- Configuration ----------------------------------------------------------
START_YEAR   = 2010
END_YEAR     = 2024
AREA         = [37, 68, 6, 98]  # North, West, South, East
RESOLUTION   = "0.25/0.25"
VARIABLE     = "total_precipitation"
PRODUCT_TYPE = "reanalysis"
DATASET_NAME = "reanalysis-era5-single-levels"

ROOT        = pathlib.Path(__file__).resolve().parent
YEARLY_DIR  = ROOT / "data" / "era5_extended" / "yearly"
MERGED_FILE = ROOT / "era5_tp_india_2010_2024.nc"
STATUS_FILE = ROOT / "ERA5_EXTENDED_DOWNLOAD_STATUS.md"

MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds between retries

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]
TIMES  = [f"{h:02d}:00" for h in range(24)]


# ---- Credentials ------------------------------------------------------------

def load_client():
    """Load CDS API client from .env. Never prints the key."""
    load_dotenv()
    import cdsapi

    url = os.environ.get("CDS_API_URL", "")
    key = os.environ.get("CDS_API_KEY", "")

    if not url or not key or key == "YOUR_CDS_API_KEY_HERE":
        print("ERROR: CDS_API_URL or CDS_API_KEY not configured in .env")
        sys.exit(1)
    if "cds-beta" in url:
        print("ERROR: CDS_API_URL still points to the decommissioned beta endpoint.")
        print("       Update to: https://cds.climate.copernicus.eu/api")
        sys.exit(1)

    return cdsapi.Client(url=url, key=key)


# ---- NetCDF validation ------------------------------------------------------

def validate_nc(path: pathlib.Path, year: int) -> tuple[bool, str]:
    """
    Returns (ok: bool, message: str).
    Checks: file exists, size > 1MB, opens as NetCDF, has 'tp', correct time range.
    """
    try:
        import xarray as xr
        import numpy as np

        if not path.exists():
            return False, "File does not exist"

        size_mb = path.stat().st_size / 1_048_576
        if size_mb < 1.0:
            return False, f"File too small ({size_mb:.2f} MB) — likely corrupt or empty"

        ds = xr.open_dataset(path)
        if "tp" not in ds.data_vars:
            ds.close()
            return False, "Variable 'tp' not found in file"

        time_var = ds.valid_time if "valid_time" in ds.coords else ds.time
        years_in_file = set(int(t) for t in time_var.dt.year.values)
        ds.close()

        if year not in years_in_file:
            return False, f"Year {year} not in file timestamps (found: {sorted(years_in_file)})"

        return True, f"OK ({size_mb:.1f} MB, {len(time_var.values)} timesteps)"

    except Exception as e:
        return False, f"Validation error: {e}"


# ---- Download one year ------------------------------------------------------

def download_year(client, year: int) -> tuple[bool, str]:
    """
    Download a single year. Returns (success, message).
    Retries on network failures, not on API 403/400 errors.
    """
    out_path = YEARLY_DIR / f"era5_tp_india_{year}.nc"

    # Skip if already valid
    ok, msg = validate_nc(out_path, year)
    if ok:
        print(f"  [{year}] SKIP — already valid: {msg}")
        return True, msg

    print(f"  [{year}] Requesting from CDS API...")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(
                DATASET_NAME,
                {
                    "product_type": PRODUCT_TYPE,
                    "variable": VARIABLE,
                    "year": str(year),
                    "month": MONTHS,
                    "day": DAYS,
                    "time": TIMES,
                    "area": AREA,
                    "format": "netcdf",
                },
                str(out_path),
            )
            # Validate after download
            ok, msg = validate_nc(out_path, year)
            if ok:
                print(f"  [{year}] DONE — {msg}")
                return True, msg
            else:
                print(f"  [{year}] Downloaded but failed validation: {msg}")
                return False, msg

        except KeyboardInterrupt:
            print(f"\n  [{year}] Interrupted by user.")
            raise

        except Exception as e:
            err_str = str(e)
            is_permanent = any(x in err_str.lower() for x in ["403", "forbidden", "too large", "cost limits"])
            print(f"  [{year}] Attempt {attempt}/{MAX_RETRIES} FAILED: {err_str[:120]}")
            if is_permanent or attempt == MAX_RETRIES:
                return False, err_str[:200]
            print(f"  [{year}] Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    return False, "Max retries exceeded"


# ---- Merge ------------------------------------------------------------------

def merge_all(required_years: list[int]) -> tuple[bool, str]:
    """
    Merge all yearly files into one chronological NetCDF.
    Verifies consistency before merging.
    """
    import xarray as xr
    import numpy as np

    print("\nMerge step:")

    paths = []
    missing = []
    for y in required_years:
        p = YEARLY_DIR / f"era5_tp_india_{y}.nc"
        ok, msg = validate_nc(p, y)
        if ok:
            paths.append(p)
        else:
            missing.append(y)

    if missing:
        return False, f"Cannot merge — missing/invalid years: {missing}"

    print(f"  Opening {len(paths)} yearly files...")
    ds_list = [xr.open_dataset(p) for p in paths]

    # Consistency checks
    ref_lat = ds_list[0].latitude.values
    ref_lon = ds_list[0].longitude.values
    for i, ds in enumerate(ds_list[1:], 1):
        if not np.allclose(ds.latitude.values, ref_lat, atol=0.001):
            [d.close() for d in ds_list]
            return False, f"Latitude mismatch in file {i}"
        if not np.allclose(ds.longitude.values, ref_lon, atol=0.001):
            [d.close() for d in ds_list]
            return False, f"Longitude mismatch in file {i}"

    print("  Grid consistency: PASSED")
    print("  Concatenating along time...")

    merged = xr.concat(ds_list, dim="valid_time")

    # Check chronological + no duplicates
    times = merged.valid_time.values
    if len(times) != len(set(t.item() for t in times)):
        [d.close() for d in ds_list]
        merged.close()
        return False, "Duplicate timestamps found in merged dataset"

    print("  No duplicate timestamps: PASSED")
    print(f"  Saving to {MERGED_FILE.name}...")
    merged.to_netcdf(str(MERGED_FILE))

    [d.close() for d in ds_list]
    merged.close()

    size_gb = MERGED_FILE.stat().st_size / 1_073_741_824
    return True, f"Merged file: {MERGED_FILE.name} ({size_gb:.2f} GB)"


# ---- Verify -----------------------------------------------------------------

def verify_all(required_years: list[int]) -> dict:
    """Check which years exist and are valid."""
    results = {}
    for y in required_years:
        p = YEARLY_DIR / f"era5_tp_india_{y}.nc"
        ok, msg = validate_nc(p, y)
        results[y] = {"ok": ok, "msg": msg, "path": str(p)}
    return results


# ---- Status document --------------------------------------------------------

def write_status(results: dict, merge_status: str | None = None):
    completed = [y for y, r in results.items() if r["ok"]]
    failed    = [y for y, r in results.items() if not r["ok"]]

    total_size = sum(
        (YEARLY_DIR / f"era5_tp_india_{y}.nc").stat().st_size
        for y in completed
        if (YEARLY_DIR / f"era5_tp_india_{y}.nc").exists()
    )
    size_gb = total_size / 1_073_741_824

    merged_exists = MERGED_FILE.exists()
    merged_size   = f"{MERGED_FILE.stat().st_size / 1_073_741_824:.2f} GB" if merged_exists else "N/A"

    lines = [
        "# ERA5 Extended Download Status",
        f"\n_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "## Request Parameters",
        "| Field | Value |",
        "|---|---|",
        f"| Dataset | {DATASET_NAME} |",
        f"| Variable | {VARIABLE} (`tp`) |",
        f"| Period | {START_YEAR}–{END_YEAR} |",
        f"| Domain | N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]} (India) |",
        f"| Resolution | 0.25° × 0.25° |",
        f"| Time | Hourly (all months, days, hours) |",
        f"| Chunking strategy | One calendar year per API request |",
        "",
        "## Progress",
        f"- **Completed years** ({len(completed)}): {completed or 'None'}",
        f"- **Failed/missing years** ({len(failed)}): {failed or 'None'}",
        f"- **Total downloaded size**: {size_gb:.2f} GB",
        "",
        "## Per-Year Status",
        "| Year | Status | Detail |",
        "|---|---|---|",
    ]
    for y, r in sorted(results.items()):
        status = "✅ OK" if r["ok"] else "❌ MISSING/FAILED"
        lines.append(f"| {y} | {status} | {r['msg']} |")

    lines += [
        "",
        "## Merged File",
        f"- **Path**: `era5_tp_india_2010_2024.nc`",
        f"- **Exists**: {'Yes' if merged_exists else 'No'}",
        f"- **Size**: {merged_size}",
        f"- **Status**: {merge_status or ('Not yet run' if not merged_exists else 'Exists')}",
        "",
        "## Commands",
        "```bash",
        "# Resume download (skips completed years automatically):",
        "python download_era5_chunked.py",
        "",
        "# Download a specific year only:",
        "python download_era5_chunked.py --year 2015",
        "",
        "# Verify all downloaded years:",
        "python download_era5_chunked.py --verify-only",
        "",
        "# Merge after all years complete:",
        "python download_era5_chunked.py --merge-only",
        "```",
        "",
        "## Known CDS Limitations",
        "- Hourly 15-year requests are rejected by CDS with `cost limits exceeded`.",
        "- Solution: one calendar year per request (implemented here).",
        "- Each year ~700 MB–1.5 GB depending on hourly density.",
        "- CDS queues requests; download may be slow during peak hours.",
        "- Yearly files are kept as recovery checkpoints after merging.",
    ]

    STATUS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Status document written: {STATUS_FILE.name}")


# ---- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Resumable ERA5 chunked downloader")
    parser.add_argument("--year",        type=int, help="Download a specific year only")
    parser.add_argument("--merge-only",  action="store_true", help="Skip download; merge existing files")
    parser.add_argument("--verify-only", action="store_true", help="Check which years are valid")
    args = parser.parse_args()

    YEARLY_DIR.mkdir(parents=True, exist_ok=True)
    required_years = list(range(START_YEAR, END_YEAR + 1))

    print("=" * 65)
    print("  ERA5 CHUNKED DOWNLOADER")
    print("=" * 65)
    print(f"  Period     : {START_YEAR} – {END_YEAR}")
    print(f"  Domain     : N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]}")
    print(f"  Variable   : {VARIABLE}")
    print(f"  Output dir : {YEARLY_DIR}")
    print()

    if args.verify_only:
        print("Verify-only mode:")
        results = verify_all(required_years)
        for y, r in sorted(results.items()):
            flag = "OK    " if r["ok"] else "MISSING"
            print(f"  {y}: [{flag}] {r['msg']}")
        write_status(results)
        return

    if args.merge_only:
        results = verify_all(required_years)
        ok, msg = merge_all(required_years)
        print(f"  Merge result: {'PASS' if ok else 'FAIL'} — {msg}")
        write_status(results, merge_status=msg)
        return

    # --- Download loop ---
    client = load_client()

    if args.year:
        target_years = [args.year]
    else:
        target_years = required_years

    session_results = {}
    for year in target_years:
        ok, msg = download_year(client, year)
        session_results[year] = {"ok": ok, "msg": msg}

    # Full verify for status doc
    results = verify_all(required_years)
    write_status(results)

    # Final summary
    print()
    print("=" * 65)
    completed = [y for y, r in results.items() if r["ok"]]
    missing   = [y for y, r in results.items() if not r["ok"]]
    total_size = sum(
        (YEARLY_DIR / f"era5_tp_india_{y}.nc").stat().st_size
        for y in completed
        if (YEARLY_DIR / f"era5_tp_india_{y}.nc").exists()
    )
    print(f"  Completed years : {completed}")
    print(f"  Missing years   : {missing}")
    print(f"  Total size      : {total_size / 1_073_741_824:.2f} GB")
    print(f"  Merged file     : {'EXISTS' if MERGED_FILE.exists() else 'NOT YET (run --merge-only after all years done)'}")
    print("=" * 65)

    if not missing:
        print("\nAll years downloaded! Run merge step:")
        print("  python download_era5_chunked.py --merge-only")


if __name__ == "__main__":
    main()
