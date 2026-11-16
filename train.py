import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import DualHeadDisasterPINN

def train():
    os.makedirs("weights", exist_ok=True)
    csv_path = os.path.join("data", "disaster_aware_forecasting_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    forecast_features = [
        "lag1_w_orographic", "lag1_specific_humidity_q", "lag1_moisture_flux",
        "lag1_dewpoint_depression", "lag1_temp_c", "lag1_rel_humidity",
        "lag1_u_wind", "lag1_v_wind", "soil_saturation_api"
    ]

    target_total = "precipitation_mm"
    target_imax = "peak_intensity_mm_hr"
    target_alert = "alert_tier"

    # Strict temporal partition: train on 2015-2020, test on unseen future 2021-2023
    train_mask = df["date"].dt.year <= 2020
    test_mask = df["date"].dt.year > 2020

    means = df.loc[train_mask, forecast_features].mean().values.astype(np.float32)
    stds = df.loc[train_mask, forecast_features].std().values.astype(np.float32)
    stds[stds == 0] = 1.0

    X_train = (df.loc[train_mask, forecast_features].values - means) / stds
    y_train_tot = df.loc[train_mask, target_total].values.astype(np.float32)
    y_train_imax = df.loc[train_mask, target_imax].values.astype(np.float32)
    c_train_lag = df.loc[train_mask, "lag1_C_oro"].values.astype(np.float32)

    X_test = (df.loc[test_mask, forecast_features].values - means) / stds
    y_test_tot = df.loc[test_mask, target_total].values.astype(np.float32)
    y_test_imax = df.loc[test_mask, target_imax].values.astype(np.float32)
    y_test_alert = df.loc[test_mask, target_alert].values.astype(np.int64)
    c_test_lag = df.loc[test_mask, "lag1_C_oro"].values.astype(np.float32)
    api_test = df.loc[test_mask, "soil_saturation_api"].values.astype(np.float32)

    seeds = [42, 101, 2024]
    state_dicts = []
    test_preds_tot = []
    test_preds_imax = []

    print(f"Training 3-seed Disaster PINN ensemble on {len(X_train)} samples...")
    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train_tot = torch.tensor(y_train_tot, dtype=torch.float32).unsqueeze(-1)
    t_y_train_imax = torch.tensor(y_train_imax, dtype=torch.float32).unsqueeze(-1)
    t_C_train = torch.tensor(c_train_lag, dtype=torch.float32).unsqueeze(-1)

    t_X_test = torch.tensor(X_test, dtype=torch.float32)
    t_C_test = torch.tensor(c_test_lag, dtype=torch.float32).unsqueeze(-1)

    for seed in seeds:
        torch.manual_seed(seed)
        model = DualHeadDisasterPINN(in_f=len(forecast_features))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
        loss_fn = nn.HuberLoss(delta=1.5)

        for epoch in range(380):
            optimizer.zero_grad()
            p_tot, p_imax, eff = model(t_X_train, t_C_train)

            l_tot = loss_fn(p_tot, t_y_train_tot)
            l_imax = loss_fn(p_imax, t_y_train_imax)

            err_tot = t_y_train_tot - p_tot
            err_imax = t_y_train_imax - p_imax
            l_miss = (
                torch.mean(torch.relu(err_tot) * (t_y_train_tot >= 45.0).float() * 2.5) +
                torch.mean(torch.relu(err_imax) * (t_y_train_imax >= 15.0).float() * 3.0)
            )

            w_oro_lag = t_X_train[:, 0:1]
            l_shadow = torch.mean(eff[w_oro_lag < 0]**2)

            total_loss = l_tot + 1.2 * l_imax + 0.2 * l_miss + 0.05 * l_shadow
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pt, pi, _ = model(t_X_test, t_C_test)
            test_preds_tot.append(pt.numpy().flatten())
            test_preds_imax.append(pi.numpy().flatten())

        state_dicts.append(model.state_dict())

    # Ensemble evaluation
    ens_p_tot = np.mean(test_preds_tot, axis=0)
    ens_p_imax = np.mean(test_preds_imax, axis=0)

    # Operational Disaster Alert Classification
    pred_red = (
        (ens_p_imax >= 18.0) |
        (ens_p_tot >= 65.0) |
        ((ens_p_tot >= 38.0) & (api_test >= 38.0))
    ).astype(int)

    actual_red = (y_test_alert == 2).astype(int)
    hits = np.sum((actual_red == 1) & (pred_red == 1))
    misses = np.sum((actual_red == 1) & (pred_red == 0))
    false_alarms = np.sum((actual_red == 0) & (pred_red == 1))

    csi = hits / (hits + misses + false_alarms)
    pod = hits / (hits + misses)
    far = false_alarms / (hits + false_alarms)

    ss_res = np.sum((y_test_tot - ens_p_tot)**2)
    ss_tot = np.sum((y_test_tot - np.mean(y_test_tot))**2)
    r2_tot = 1.0 - (ss_res / ss_tot)
    mae_tot = np.mean(np.abs(y_test_tot - ens_p_tot))

    ss_res_i = np.sum((y_test_imax - ens_p_imax)**2)
    ss_tot_i = np.sum((y_test_imax - np.mean(y_test_imax))**2)
    r2_imax = 1.0 - (ss_res_i / ss_tot_i)

    print("==========================================================================")
    print(" 24-HOUR FORECAST TEST SET BENCHMARK (2021-2023)")
    print("==========================================================================")
    print(f"Red Alert CSI (Threat Score) : {csi:.4f}")
    print(f"Probability of Detection (POD) : {pod*100:.1f}% ({hits}/{hits+misses})")
    print(f"Fatal Missed Catastrophes    : {misses}")
    print(f"False Alarm Ratio (FAR)       : {far*100:.1f}% ({false_alarms})")
    print(f"Total Rainfall R^2           : {r2_tot:.4f}")
    print(f"Total Rainfall MAE           : {mae_tot:.4f} mm")
    print(f"Peak Hourly Intensity R^2     : {r2_imax:.4f}")
    print("==========================================================================")

    checkpoint = {
        "seeds": seeds,
        "state_dicts": state_dicts,
        "features": forecast_features,
        "means": means,
        "stds": stds,
        "metrics": {
            "csi": float(csi),
            "pod": float(pod),
            "far": float(far),
            "r2_tot": float(r2_tot),
            "mae_tot": float(mae_tot),
            "r2_imax": float(r2_imax)
        }
    }

    out_file = os.path.join("weights", "pinn_ensemble.pt")
    torch.save(checkpoint, out_file)
    print(f"Trained weights saved to {out_file}")

if __name__ == "__main__":
    train()
