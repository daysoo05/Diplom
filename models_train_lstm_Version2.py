#!/usr/bin/env python3
"""
Train a simple LSTM for next-day temperature forecasting on a univariate series (t2m_c).

Usage:
  python models/train_lstm.py --in data/processed/series_berlin.csv --out models/model.pt --epochs 30
"""
import argparse
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_dataset(series, window):
    X, y = [], []
    for i in range(len(series)-window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X)[:,:,None].astype(np.float32)
    y = np.array(y).astype(np.float32)[:,None]
    return X, y

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="csv_in", required=True)
    p.add_argument("--out", default="models/model.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--window", type=int, default=14)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

def main(csv_in, out_model, epochs, window, lr):
    df = pd.read_csv(csv_in, parse_dates=['time'])
    if 't2m_c' not in df.columns:
        raise RuntimeError("Input CSV must contain t2m_c column produced by preprocess.py")
    series = df['t2m_c'].fillna(method='ffill').values
    mean, std = series.mean(), series.std()
    series_norm = (series - mean) / (std + 1e-8)
    X, y = make_dataset(series_norm, window)
    split = int(len(X)*0.8)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    model = LSTMForecast()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb), yb).item()
        val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{epochs}, val_loss={val_loss:.4f}")
    os.makedirs(os.path.dirname(out_model) or ".", exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'mean': float(mean), 'std': float(std), 'window': window}, out_model)
    print(f"Saved model to {out_model}")

if __name__ == "__main__":
    args = parse_args()
    main(args.csv_in, args.out, args.epochs, args.window, args.lr)