# Climate Monitor — Minimal Prototype

This project monitors climate changes using public servers (ERA5 via Copernicus CDS), preprocesses data, trains a simple forecasting model, and offers a Streamlit dashboard.

Purpose
- Demonstrate an automated pipeline to ingest public climate reanalysis (ERA5), extract meaningful series, train a small ML model, and visualize forecasts.
- Provide a foundation you can expand (spatial models, ensembles, additional data sources, cloud deployment).

Key features
- ERA5 ingestion (CDS API)
- xarray-based preprocessing and resampling
- PyTorch LSTM model for short-term forecasting (7-day)
- Streamlit dashboard to view observations and forecasts
- Docker + docker-compose for local testing
- GitHub Actions workflow (example) to run scheduled ingestion and training

Default configuration
- Default point: Berlin (lat: 52.52, lon: 13.405)
- Default variables: 2m temperature (t2m) and total precipitation (tp)
- Forecast horizon: 7 days
- Update cadence: daily (workflow example)

Quick start (local)
1. Install Python 3.9+ and create a venv:
   python -m venv venv && source venv/bin/activate
2. Install requirements:
   pip install -r requirements.txt
3. Configure Copernicus CDS API (for ERA5 ingestion):
   - Create ~/.cdsapirc with credentials (see https://cds.climate.copernicus.eu/api-how-to)
4. Run a small demo pipeline (example):
   - python scripts/ingest_era5.py --start 2022-01-01 --end 2022-03-31 --out data/raw/era5_2022_Q1.nc --variables t2m tp --bbox 49 5 55 20
   - python scripts/preprocess.py --in data/raw/era5_2022_Q1.nc --lat 52.52 --lon 13.405 --out data/processed/series_berlin.csv
   - python models/train_lstm.py --in data/processed/series_berlin.csv --out models/model.pt --epochs 30
   - streamlit run serving/streamlit_app.py
5. Open http://localhost:8501 to see the dashboard.

Docker (local)
- Build & run:
  docker-compose up --build
- The app will be available at http://localhost:8501 (ensure data and credentials are mounted or available inside container if you want ingestion to run from there).

Configuration
- configs/config.yaml contains defaults. Modify lat/lon, bbox, variables, start/end dates, and horizon.

Notes & next steps
- This is a minimal prototype: production systems require robust error handling, retries, storage (S3), models for spatial grids (ConvLSTM/Transformers), probabilistic ensembles, monitoring and alerting, and careful evaluation/calibration.
- If you want, I can:
  - Push these files to a branch in your repo once you add an initial commit.
  - Extend to spatial forecasts (ConvLSTM), add ensemble uncertainty, or integrate additional data sources (NOAA, NASA, GFS).
  - Prepare step-by-step video/text walkthrough for non-coders.

License
- MIT by default (see LICENSE file).