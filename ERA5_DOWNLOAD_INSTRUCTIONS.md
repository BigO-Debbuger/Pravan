# ERA5 Extended Dataset Download Instructions

To extend our ML models (CNN / ConvLSTM) in `EXP-005` and `EXP-006`, we require a much larger historical dataset than the current 28 months (2020–2022). The target is **2010–2024 (15 years)**.

Due to the size and authentication required by the Copernicus Climate Data Store (CDS), this process must be run locally with user credentials.

## 1. Setup Your CDS API Credentials Safely

We use a `.env` file to securely store your credentials without committing them to Git.

1. Register / Log in to the Copernicus Data Space Ecosystem (CDS Beta):
   https://cds-beta.climate.copernicus.eu/

2. Go to your user profile page to get your Personal Access Token.

3. Open the `.env` file in the project root directory. (If it doesn't exist, copy `.env.example` to `.env`).

4. Add your personal access token to the `CDS_API_KEY` variable:
   ```env
   CDS_API_URL=https://cds.climate.copernicus.eu/api
   CDS_API_KEY=YOUR_ACTUAL_KEY_HERE
   ```
   > **IMPORTANT:** The old `cds-beta.climate.copernicus.eu` URL was decommissioned on September 26, 2024.
   > You must use `https://cds.climate.copernicus.eu/api` (no `-beta`).
   > Using the old URL causes an SSL certificate error because the server no longer exists.
   
   **DO NOT share or commit this `.env` file.** (It is already added to `.gitignore`).

## 2. Install Required Python Packages

```bash
pip install cdsapi python-dotenv
```

## 3. Test Authentication

Before starting the massive download, verify your configuration safely:

```bash
python download_era5_extended.py --test
```

## 4. Run the Download Script

The script is pre-configured to match the existing India domain (37N-6S, 68E-98E) and spatial resolution (0.25° × 0.25°).

```bash
python download_era5_extended.py
```

## IMPORTANT: Data Size Warning
The request asks for 15 years of hourly data. This will result in a very large NetCDF file (potentially ~5-10 GB). 
- If the CDS API rejects the request for being too large, you may need to edit `download_era5_extended.py` to loop over years and download them into separate files, then merge them using `xarray`.
- Keep this downloaded data **excluded from Git**.

## What To Do After Downloading

Once the download is successful (e.g., `era5_tp_india_2010_2024.nc` is created):
1. Rename the old dataset folder/file if necessary.
2. Update `calculate_climatology.py` and `ml/preprocessing.py` to point to the new filename.
3. Rerun `python calculate_climatology.py` to generate the 15-year climatology and anomalies.
4. Rerun `python ml/features.py` to regenerate the parquet datasets.
5. You are then unblocked to start `EXP-005`.
