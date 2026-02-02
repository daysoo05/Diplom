#!/usr/bin/env python3
"""
Load a trained model and produce a multi-day forecast using iterative single-step predictions.

Usage:
  python models/predict.py --model models/model.pt --in data/processed/series_berlin.csv --days 7
"""
import argparse
import pandas as pd
import numpy as np
import torch
from models.train_lstm import LSTMForecast

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--in", required=True)
    p.add_argument("--days", type=int, default=7)
    return p.parse_args()

def main(model_path, csv_in, days):
    import torch
    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint['model']
    mean, std = checkpoint['mean'], checkpoint['std']
    window = checkpoint.get('window', 14)
    model = LSTMForecast()
    model.load_state_dict(state)
    model.eval()

    df = pd.read_csv(csv_in, parse_dates=['time'])
    series = df['t2m_c'].fillna(method='ffill').values
    norm = (series - mean) / (std + 1e-8)
    past = norm[-window:].tolist()
    preds = []
    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor(np.array(past[-window:])[None,:,None], dtype=torch.float32)
            p = model(x).cpu().numpy().ravel()[0]
            preds.append(p * std + mean)
            past.append(p)
    last_date = pd.to_datetime(df['time'].iloc[-1])
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
    out = pd.DataFrame({"time": future_dates, "t2m_c_pred": preds})
    print(out.to_csv(index=False))
    return out

if __name__ == "__main__":
    args = parse_args()
    main(args.model, args.in, args.days)