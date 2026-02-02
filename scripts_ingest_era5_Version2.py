#!/usr/bin/env python3
"""
Ingest ERA5 data (small region or full) using the Copernicus CDS API.

Usage:
  python scripts/ingest_era5.py --start 2022-01-01 --end 2022-01-31 --out data/raw/era5_jan2022.nc --variables t2m tp --bbox 49 5 55 20

Notes:
- Requires `cdsapi` and a configured ~/.cdsapirc file (https://cds.climate.copernicus.eu/api-how-to)
- This script requests monthly data by year/month. For simplicity we take the date window and request months that intersect.
"""
import argparse
import cdsapi
import os
from datetime import datetime, timedelta
import calendar

VAR_MAP = {
    "t2m": "2m_temperature",        # variable name mapping
    "tp": "total_precipitation",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--variables", nargs="+", default=["t2m"], help="Variables short names: t2m tp")
    p.add_argument("--bbox", nargs=4, type=float, metavar=('lat_min','lon_min','lat_max','lon_max'),
                   help="Optional bounding box (lat_min lon_min lat_max lon_max) — for smaller downloads")
    p.add_argument("--out", default="data/raw/era5.nc", help="Output NetCDF path")
    return p.parse_args()

def months_between(start_date, end_date):
    months = []
    cur = datetime(start_date.year, start_date.month, 1)
    while cur <= end_date:
        months.append((cur.year, f"{cur.month:02d}"))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    return months

def build_request(year, month, vars, bbox):
    variables = []
    for v in vars:
        if v not in VAR_MAP:
            raise ValueError(f"Unknown variable {v}")
        variables.append(VAR_MAP[v])
    req = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": variables,
        "year": [str(year)],
        "month": [str(month)],
        "day": [f"{d:02d}" for d in range(1,32)],
        "time": [f"{h:02d}:00" for h in range(0,24)],
        # area: north, west, south, east
    }
    if bbox:
        lat_min, lon_min, lat_max, lon_max = bbox
        # convert to CDS area ordering: north, west, south, east
        req["area"] = [lat_max, lon_min, lat_min, lon_max]
    return req

def download_month(client, year, month, vars, bbox, out_path):
    req = build_request(year, month, vars, bbox)
    tmp_out = out_path + f".part.{year}{month}"
    print(f"Requesting ERA5 {year}-{month} -> {tmp_out}")
    client.retrieve('reanalysis-era5-single-levels', req, tmp_out)
    # If out_path exists, we will concatenate/merge later. For simplicity we save one file per run.
    # Move tmp_out to final path if not exists
    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        os.rename(tmp_out, out_path)
    else:
        # simple concat: append to history name (user can handle combining with xarray open_mfdataset)
        new_name = out_path.replace(".nc", f".{year}{month}.nc")
        os.rename(tmp_out, new_name)
    print("Download finished.")

def main():
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    months = months_between(start, end)
    client = cdsapi.Client()
    for (y,m) in months:
        download_month(client, y, m, args.variables, args.bbox, args.out)

if __name__ == "__main__":
    main()