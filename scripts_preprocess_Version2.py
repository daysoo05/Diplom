#!/usr/bin/env python3
"""
Preprocess ERA5 NetCDF file:
- Load variables
- Select nearest gridpoint to lat/lon (or average small area)
- Convert units when needed
- Compute daily aggregated series and save CSV.

Usage:
  python scripts/preprocess.py --in data/raw/era5.nc --lat 52.52 --lon 13.405 --out data/processed/series_berlin.csv
"""
import argparse
import xarray as xr
import pandas as pd
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="nc_in", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--out", default="data/processed/series.csv")
    p.add_argument("--agg", choices=["point","mean"], default="point", help="point: nearest gridpoint; mean: mean over small region")
    return p.parse_args()

def extract_series(nc_in, lat, lon, out_csv, agg="point"):
    ds = xr.open_dataset(nc_in)
    # Common ERA5 variable names: t2m (Kelvin) and tp (m)
    varnames = [v for v in ("t2m","tp") if v in ds]
    if not varnames:
        raise RuntimeError(f"No expected variables found. Found: {list(ds.data_vars)}")
    # Build dataframe with time, t2m_c, tp_mm
    records = []
    for var in varnames:
        if agg == "point":
            sel = ds[var].sel(latitude=lat, longitude=lon, method="nearest")
        else:
            # mean over a small area +- 0.25 deg
            sel = ds[var].sel(latitude=slice(lat+0.25, lat-0.25), longitude=slice(lon-0.25, lon+0.25)).mean(dim=("latitude","longitude"))
        df = sel.to_dataframe().reset_index()
        df = df.rename(columns={var: var})
        if records:
            records[0] = records[0].merge(df[['time',var]], on='time', how='outer')
        else:
            records.append(df[['time',var]])
    merged = records[0]
    merged['time'] = pd.to_datetime(merged['time'])
    # Convert units: t2m K -> C ; tp m -> mm (daily accum)
    if 't2m' in merged:
        merged['t2m_c'] = merged['t2m'] - 273.15
    if 'tp' in merged:
        # ERA5 tp is accumulated precipitation in meters for each timestep; convert to mm
        merged['tp_mm'] = merged['tp'] * 1000.0
    # Resample to daily aggregates
    merged = merged.set_index('time').resample('D').agg({
        't2m_c': 'mean' if 't2m_c' in merged else None,
        'tp_mm': 'sum' if 'tp_mm' in merged else None
    }).reset_index()
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"Wrote processed series to {out_csv}")

if __name__ == "__main__":
    args = parse_args()
    extract_series(args.nc_in, args.lat, args.lon, args.out, args.agg)