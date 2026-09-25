# ERA5 Extended Download Status

_Last updated: 2026-09-25 00:07:53_

## Request Parameters
| Field | Value |
|---|---|
| Dataset | reanalysis-era5-single-levels |
| Variable | total_precipitation (`tp`) |
| Period | 2010–2024 |
| Domain | N=37 W=68 S=6 E=98 (India) |
| Resolution | 0.25° × 0.25° |
| Time | Hourly (all months, days, hours) |
| Chunking strategy | One calendar year per API request |

## Progress
- **Completed years** (15): [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
- **Failed/missing years** (0): None
- **Total downloaded size**: 1.64 GB

## Per-Year Status
| Year | Status | Detail |
|---|---|---|
| 2010 | ✅ OK | OK (113.9 MB, 8760 timesteps) |
| 2011 | ✅ OK | OK (114.2 MB, 8760 timesteps) |
| 2012 | ✅ OK | OK (111.4 MB, 8784 timesteps) |
| 2013 | ✅ OK | OK (113.1 MB, 8760 timesteps) |
| 2014 | ✅ OK | OK (109.3 MB, 8760 timesteps) |
| 2015 | ✅ OK | OK (112.8 MB, 8760 timesteps) |
| 2016 | ✅ OK | OK (107.1 MB, 8784 timesteps) |
| 2017 | ✅ OK | OK (114.0 MB, 8760 timesteps) |
| 2018 | ✅ OK | OK (111.7 MB, 8760 timesteps) |
| 2019 | ✅ OK | OK (113.1 MB, 8760 timesteps) |
| 2020 | ✅ OK | OK (112.6 MB, 8784 timesteps) |
| 2021 | ✅ OK | OK (115.2 MB, 8760 timesteps) |
| 2022 | ✅ OK | OK (116.0 MB, 8760 timesteps) |
| 2023 | ✅ OK | OK (108.9 MB, 8760 timesteps) |
| 2024 | ✅ OK | OK (111.0 MB, 8784 timesteps) |

## Merged File
- **Path**: `era5_tp_india_2010_2024.nc`
- **Exists**: Yes
- **Size**: 1.61 GB
- **Status**: Merged file: era5_tp_india_2010_2024.nc (1.61 GB)

## Commands
```bash
# Resume download (skips completed years automatically):
python download_era5_chunked.py

# Download a specific year only:
python download_era5_chunked.py --year 2015

# Verify all downloaded years:
python download_era5_chunked.py --verify-only

# Merge after all years complete:
python download_era5_chunked.py --merge-only
```

## Known CDS Limitations
- Hourly 15-year requests are rejected by CDS with `cost limits exceeded`.
- Solution: one calendar year per request (implemented here).
- Each year ~700 MB–1.5 GB depending on hourly density.
- CDS queues requests; download may be slow during peak hours.
- Yearly files are kept as recovery checkpoints after merging.