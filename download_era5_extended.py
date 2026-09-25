"""
download_era5_extended.py
-------------------------
Downloads extended ERA5 precipitation data from 2010 to 2024.
Requires the 'cdsapi' and 'python-dotenv' python packages.
See ERA5_DOWNLOAD_INSTRUCTIONS.md for setup details.

Domain: India
- North: 37.0
- West: 68.0
- South: 6.0
- East: 98.0
Resolution: 0.25 deg x 0.25 deg
Variable: Total Precipitation (tp)
"""

import cdsapi
import os
import sys
from dotenv import load_dotenv

# NOTE: We do NOT suppress SSL warnings here.
# If SSL warnings appear, they indicate a real configuration problem to fix, not ignore.

def get_cds_client():
    """
    Initializes the CDS API client safely using credentials from .env.
    
    Priority of credential resolution (highest to lowest):
      1. Values explicitly passed to cdsapi.Client(url=..., key=...)  <- we use this
      2. ~/.cdsapirc file (exists but is empty on this machine)
      3. CDSAPI_URL / CDSAPI_KEY environment variables
    
    By explicitly passing url= and key= we bypass ~/.cdsapirc entirely.
    """
    load_dotenv()  # Loads .env into os.environ

    url = os.environ.get("CDS_API_URL")
    key = os.environ.get("CDS_API_KEY")

    if not url:
        print("ERROR: CDS_API_URL is missing or not set in .env")
        print("Please read ERA5_DOWNLOAD_INSTRUCTIONS.md to configure credentials.")
        sys.exit(1)

    if not key or key == "YOUR_CDS_API_KEY_HERE":
        print("ERROR: CDS_API_KEY is missing or still a placeholder in .env")
        print("Please read ERA5_DOWNLOAD_INSTRUCTIONS.md to configure credentials.")
        sys.exit(1)

    # Validate URL points to the live production endpoint
    if "cds-beta" in url:
        print("ERROR: Your CDS_API_URL still points to the decommissioned beta endpoint.")
        print("  Current:  ", url)
        print("  Required: https://cds.climate.copernicus.eu/api")
        print("Please update CDS_API_URL in your .env file.")
        sys.exit(1)

    # Explicitly pass url and key so the cdsapi client does NOT fall back to ~/.cdsapirc
    return cdsapi.Client(url=url, key=key)


def test_auth():
    """Tests authentication configuration without downloading any data."""
    print("Testing CDS API authentication...")
    print()

    # Show which .env is being used
    import dotenv
    env_path = dotenv.find_dotenv()
    print(f"  .env file loaded from: {env_path if env_path else '(none found)'}")

    load_dotenv()
    url = os.environ.get("CDS_API_URL", "(not set)")
    print(f"  CDS_API_URL (from .env): {url}")
    print(f"  CDS_API_KEY:             [REDACTED - not shown]")
    print()

    # Confirm production URL
    if "cds-beta" in url:
        print("FAIL: URL still points to the decommissioned cds-beta endpoint.")
        print("      Update CDS_API_URL in .env to: https://cds.climate.copernicus.eu/api")
        sys.exit(1)

    # Check for ~/.cdsapirc and report (but never show the key)
    import pathlib
    rc_path = pathlib.Path.home() / ".cdsapirc"
    if rc_path.exists():
        rc_size = rc_path.stat().st_size
        print(f"  ~/.cdsapirc found (size={rc_size} bytes).")
        if rc_size == 0:
            print("  ~/.cdsapirc is empty - it will NOT override our explicit url/key.")
        else:
            print("  WARNING: ~/.cdsapirc is non-empty. Reading its url line only:")
            for line in rc_path.read_text(encoding='utf-8', errors='replace').splitlines():
                if 'url' in line.lower() and 'key' not in line.lower():
                    print(f"    {line.strip()}")
            print("  Our script passes url= explicitly to cdsapi.Client() so ~/.cdsapirc is bypassed.")
    else:
        print("  ~/.cdsapirc: not found (OK - not required)")

    print()

    try:
        c = get_cds_client()
        # Inspect what the client actually resolved
        # cdsapi.Client stores the url as .url attribute
        resolved_url = getattr(c, 'url', None) or getattr(c, '_url', None) or "(attribute not found)"
        print(f"  CDS Client initialized.")
        print(f"  Resolved endpoint (from client object): {resolved_url}")

        if "cds-beta" in str(resolved_url):
            print()
            print("FAIL: Client is connecting to the decommissioned cds-beta endpoint!")
            sys.exit(1)
        else:
            print()
            print("PASS: Client is using the correct production endpoint.")
            print("      Authentication configuration looks valid.")
            print("      Actual download will perform a full API key verification.")
        return True
    except SystemExit:
        raise
    except Exception as e:
        print(f"Authentication test failed: {e}")
        return False


def download_era5(output_file="era5_tp_india_2010_2024.nc"):
    c = get_cds_client()

    print(f"Starting CDS API download for extended ERA5 dataset...")
    print(f"Target: {output_file}")
    print("Time range: 2010 to 2024 (Hourly)")
    print("Domain: India (37N-6S, 68E-98E) at 0.25x0.25 resolution")

    years = [str(y) for y in range(2010, 2025)]
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    times = [f"{h:02d}:00" for h in range(24)]

    # The dataset name is 'reanalysis-era5-single-levels'
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'total_precipitation',
            'year': years,
            'month': months,
            'day': days,
            'time': times,
            'area': [37, 68, 6, 98],  # North, West, South, East
            'format': 'netcdf',
        },
        output_file
    )
    print(f"Download complete! Saved to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Test authentication configuration only (no download)")
    args = parser.parse_args()

    if args.test:
        test_auth()
    else:
        print("WARNING: You are about to start a very large data download (~5-10 GB).")
        print("If you only wanted to test authentication, run with --test")
        print("Starting download in 5 seconds... (Ctrl+C to cancel)")
        import time
        time.sleep(5)
        download_era5()
