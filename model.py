import numpy as np
import torch
import torch.nn as nn

class DualHeadDisasterPINN(nn.Module):
    """
    Physics-Informed Neural Network with dual output heads for bulk 24-hour rainfall
    accumulation and peak 1-hour convective burst intensity.
    """
    def __init__(self, in_f=9):
        super().__init__()
        # Shared latent representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_f, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU()
        )
        
        # Head 1: Total 24h Precipitation Efficiency E_tot in [0, 1]
        self.tot_eff_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.tot_bg = nn.Sequential(
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
        self.tot_scale = nn.Parameter(torch.tensor(1.2))
        
        # Head 2: Peak Burst Intensity Efficiency E_imax in [0, 1]
        self.imax_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.imax_scale = nn.Parameter(torch.tensor(0.35))

    def forward(self, x, c_oro_lag):
        feat = self.trunk(x)
        
        # Physics-bounded total volume: E_tot * C_oro * scale + background
        eff_tot = self.tot_eff_head(feat)
        p_tot = eff_tot * c_oro_lag * torch.relu(self.tot_scale) + torch.relu(self.tot_bg(feat)) * 0.2
        
        # Physics-bounded convective flash rate: E_imax * (0.15 * C_oro) * scale + baseline
        eff_imax = self.imax_head(feat)
        p_imax = eff_imax * (c_oro_lag * 0.15) * torch.relu(self.imax_scale) + eff_imax * 5.0
        
        return p_tot, p_imax, eff_tot

def compute_physics_features(wind_spd_kmh, wind_dir_deg, temp_c, rel_humidity, dh_dx, dh_dy, api_mm):
    """
    Couples kinematic terrain lifting with Clausius-Clapeyron saturation thermodynamics.
    """
    wind_mps = wind_spd_kmh * (1000.0 / 3600.0)
    wind_rad = np.radians(wind_dir_deg)
    u_wind = -wind_mps * np.sin(wind_rad)
    v_wind = -wind_mps * np.cos(wind_rad)
    
    # Kinematic mechanical updraft: w = u*(dh/dx) + v*(dh/dy) [m/s]
    w_oro = u_wind * dh_dx + v_wind * dh_dy
    w_pos = max(0.0, w_oro)
    
    # Specific humidity q (g/kg) via Magnus-Tetens Clausius-Clapeyron formulation
    e_sat = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))  # hPa
    e_act = (rel_humidity / 100.0) * e_sat
    q_gkg = (0.622 * e_act / (900.0 - e_act)) * 1000.0          # g/kg
    
    # Dewpoint depression (T - T_d)
    a, b = 17.27, 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(max(0.01, rel_humidity / 100.0))
    t_dew = (b * alpha) / (a - alpha)
    dewpoint_dep = max(0.0, temp_c - t_dew)
    
    # Horizontal moisture flux proxy (kg/m^2/s)
    moisture_flux = (q_gkg / 1000.0) * wind_mps
    
    # Theoretical maximum orographic condensation ceiling C_oro (mm/day)
    rho_air = 1.18  # kg/m^3
    c_oro_mm = rho_air * (q_gkg / 1000.0) * w_pos * 86400.0 * 0.05
    
    feature_vector = np.array([
        w_oro, q_gkg, moisture_flux, dewpoint_dep,
        temp_c, rel_humidity, u_wind, v_wind, api_mm
    ], dtype=np.float32)
    
    diagnostics = {
        "w_oro": w_oro,
        "q_gkg": q_gkg,
        "c_oro_mm": c_oro_mm,
        "dewpoint_dep": dewpoint_dep,
        "u_wind": u_wind,
        "v_wind": v_wind,
        "wind_mps": wind_mps
    }
    
    return feature_vector, diagnostics
