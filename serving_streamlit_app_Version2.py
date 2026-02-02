"""
Streamlit dashboard to show recent observations and a 7-day forecast.

Run:
  streamlit run serving/streamlit_app.py

The app will look for:
- data/processed/series_berlin.csv (or upload CSV)
- models/model.pt (optional)

If model is missing, a persistence baseline (yesterday's value) is shown.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
from datetime import timedelta
st.set_page_config(layout="wide", page_title="Climate Monitor")

st.title("Climate Monitor — Prototype")
st.markdown("Simple dashboard showing observations and a short-term forecast (temperature).")

uploaded = st.file_uploader("Upload processed CSV (time, t2m_c, tp_mm optional)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['time'])
else:
    default_path = "data/processed/series_berlin.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path, parse_dates=['time'])
    else:
        st.warning(f"No data found at {default_path}. Run preprocessing or upload CSV.")
        df = None

if df is not None:
    st.subheader("Recent observations")
    st.write(df.tail(10))
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['t2m_c'], label='t2m (C)')
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    st.pyplot(fig)

    # Forecast
    model_path = "models/model.pt"
    if os.path.exists(model_path):
        st.success("Model found. Generating forecast...")
        import pandas as pd
        from models.predict import main as predict_main
        # Use the predict function programmatically (it prints CSV); easiest to run it and parse output
        try:
            future = predict_main(model_path, None, 7)  # But predict_main expects file; adapt: write temp csv
        except Exception:
            # fallback to subprocess call
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            df.to_csv(tmp.name, index=False)
            cmd = ["python", "models/predict.py", "--model", model_path, "--in", tmp.name, "--days", "7"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                future = pd.read_csv(pd.compat.StringIO(res.stdout), parse_dates=['time'])
            else:
                st.error("Prediction failed: " + (res.stderr or res.stdout))
                future = None
        if future is not None:
            st.subheader("7-day forecast")
            st.write(future)
            fig2, ax2 = plt.subplots()
            ax2.plot(df['time'], df['t2m_c'], label='observed')
            ax2.plot(future['time'], future['t2m_c_pred'], '--', label='predicted')
            ax2.legend()
            st.pyplot(fig2)
    else:
        st.info("No trained model found. Showing a persistence baseline (t+1 = t).")
        last = df['t2m_c'].iloc[-1]
        future_dates = [df['time'].iloc[-1] + timedelta(days=i+1) for i in range(7)]
        baseline = pd.DataFrame({"time": future_dates, "t2m_c_pred": [last]*7})
        st.write(baseline)
        fig3, ax3 = plt.subplots()
        ax3.plot(df['time'], df['t2m_c'], label='observed')
        ax3.plot(baseline['time'], baseline['t2m_c_pred'], '--', label='persistence')
        ax3.legend()
        st.pyplot(fig3)

st.sidebar.header("Actions")
if st.sidebar.button("Run ingest (ERA5) example"):
    st.sidebar.info("This will call scripts/ingest_era5.py with a small sample configuration.")
    st.sidebar.write("Make sure ~/.cdsapirc is configured.")
    cmd = ["python", "scripts/ingest_era5.py", "--start", "2022-01-01", "--end", "2022-01-31", "--out", "data/raw/era5_jan2022.nc", "--variables", "t2m", "--bbox", "49", "5", "55", "20"]
    st.sidebar.text(" ".join(cmd))
    try:
        import subprocess, shlex
        subprocess.Popen(cmd)
        st.sidebar.success("Ingest started in background. Check logs in terminal.")
    except Exception as e:
        st.sidebar.error(str(e))