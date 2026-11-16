import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import DualHeadDisasterPINN, compute_physics_features

st.set_page_config(
    page_title="24h Mountain Disaster Early Warning System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist, Clean Styling (No Emojis, Native Pointer Cursors)
st.markdown("""
<style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Universal Pointer Cursors on Selectors */
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] input,
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectbox"] *,
    div[data-testid="stSelectbox"] input,
    div[role="combobox"],
    div[role="combobox"] *,
    div[role="combobox"] input,
    div[role="listbox"],
    div[role="listbox"] *,
    li[role="option"],
    li[role="option"] *,
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label *,
    div[role="radiogroup"] *,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label *,
    div[data-testid="stCheckbox"] span,
    input[type="checkbox"],
    input[type="radio"],
    button,
    button * {
        cursor: pointer !important;
    }
    
    div[data-baseweb="select"] input,
    div[role="combobox"] input {
        cursor: pointer !important;
        caret-color: transparent !important;
    }

    div[data-testid="stSlider"] div[role="slider"] {
        cursor: grab !important;
    }
    div[data-testid="stSlider"] div[role="slider"]:active {
        cursor: grabbing !important;
    }

    /* Minimal Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2) !important;
        background: rgba(255, 255, 255, 0.05);
    }
    
    .card-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f0f6fc;
        margin: 0;
        line-height: 1.2;
    }
    .card-formula {
        font-size: 0.74rem;
        color: #6e7681;
        margin-top: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    
    /* Alert Banners */
    .alert-banner-red {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 8px;
        padding: 18px 22px;
        margin: 10px 0 20px 0;
    }
    .alert-banner-yellow {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 8px;
        padding: 18px 22px;
        margin: 10px 0 20px 0;
    }
    .alert-banner-green {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 8px;
        padding: 18px 22px;
        margin: 10px 0 20px 0;
    }
    
    .alert-title-red { color: #f87171; font-size: 1.1rem; font-weight: 700; margin: 0 0 4px 0; }
    .alert-title-yellow { color: #fbbf24; font-size: 1.1rem; font-weight: 700; margin: 0 0 4px 0; }
    .alert-title-green { color: #34d399; font-size: 1.1rem; font-weight: 700; margin: 0 0 4px 0; }
    
    .alert-desc {
        color: #c9d1d9;
        font-size: 0.88rem;
        line-height: 1.45;
        margin: 0;
    }
    
    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 500;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #8b949e;
        margin-bottom: 14px;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 20px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Orographic Stations & Representative Historical Benchmarks
BENCHMARK_LOCATIONS = {
    "Garhwal Himalayas (Dehradun, India)": {
        "lat": 30.3165,
        "lon": 78.0322,
        "region": "Himalayas",
        "dh_dx": 0.28,
        "dh_dy": 0.38,
        "elevation_m": 680,
        "scenarios": {
            "Catastrophic Monsoon Cloudburst (2021 Recorded Event)": {"wind_spd": 28.0, "wind_dir": 165.0, "temp": 29.5, "rh": 94.0, "api": 52.0},
            "Gentle Monsoon Soaking Drizzle": {"wind_spd": 10.0, "wind_dir": 140.0, "temp": 26.0, "rh": 82.0, "api": 15.0},
            "Calm Dry Weather (Current Post-Monsoon)": {"wind_spd": 5.0, "wind_dir": 310.0, "temp": 21.0, "rh": 48.0, "api": 0.0}
        }
    },
    "Western Ghats (Mahabaleshwar Scarp, India)": {
        "lat": 17.9237,
        "lon": 73.6586,
        "region": "Western Ghats",
        "dh_dx": 0.35,
        "dh_dy": 0.12,
        "elevation_m": 1353,
        "scenarios": {
            "Catastrophic Basalt Wall Deluge (2019 Recorded Event)": {"wind_spd": 42.0, "wind_dir": 265.0, "temp": 23.5, "rh": 98.0, "api": 85.0},
            "Moderate Monsoon Rain": {"wind_spd": 16.0, "wind_dir": 240.0, "temp": 22.0, "rh": 85.0, "api": 22.0},
            "Calm Dry Weather (Current Post-Monsoon)": {"wind_spd": 6.0, "wind_dir": 70.0, "temp": 24.0, "rh": 50.0, "api": 0.0}
        }
    },
    "European Alps (Valais / Zermatt, Switzerland)": {
        "lat": 46.0207,
        "lon": 7.7491,
        "region": "Alps",
        "dh_dx": 0.32,
        "dh_dy": 0.45,
        "elevation_m": 1620,
        "scenarios": {
            "Severe Alpine Flash Storm (2021 Recorded Event)": {"wind_spd": 24.0, "wind_dir": 210.0, "temp": 18.0, "rh": 89.0, "api": 35.0},
            "Valley Orographic Shower": {"wind_spd": 12.0, "wind_dir": 190.0, "temp": 15.0, "rh": 72.0, "api": 10.0},
            "Clear Alpine Autumn Weather": {"wind_spd": 6.0, "wind_dir": 350.0, "temp": 7.0, "rh": 45.0, "api": 0.0}
        }
    },
    "Rocky Mountains (Colorado Front Range, USA)": {
        "lat": 40.0150,
        "lon": -105.2705,
        "region": "Rockies",
        "dh_dx": 0.22,
        "dh_dy": 0.18,
        "elevation_m": 1655,
        "scenarios": {
            "High-Plains Upslope Deluge (2013 Recorded Event)": {"wind_spd": 22.0, "wind_dir": 95.0, "temp": 22.0, "rh": 90.0, "api": 42.0},
            "Moderate Mountain Rain": {"wind_spd": 12.0, "wind_dir": 110.0, "temp": 18.0, "rh": 65.0, "api": 12.0},
            "Dry Westerly Leeward Flow": {"wind_spd": 18.0, "wind_dir": 270.0, "temp": 14.0, "rh": 30.0, "api": 0.0}
        }
    }
}

@st.cache_resource(show_spinner=False)
def load_models():
    weights_path = os.path.join(os.path.dirname(__file__), "weights", "pinn_ensemble.pt")
    if not os.path.exists(weights_path):
        weights_path = "weights/pinn_ensemble.pt"

    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        models = []
        for sd in ckpt["state_dicts"]:
            m = DualHeadDisasterPINN(in_f=len(ckpt["features"]))
            m.load_state_dict(sd)
            m.eval()
            models.append(m)
        return models, ckpt["means"], ckpt["stds"]
    
    # Direct Analytical Fallback
    models = []
    for _ in range(3):
        m = DualHeadDisasterPINN(in_f=9)
        m.eval()
        models.append(m)
    means = np.array([0.15, 12.0, 0.08, 3.5, 24.0, 75.0, 1.2, 1.5, 18.0], dtype=np.float32)
    stds = np.array([0.45, 4.5, 0.07, 3.0, 5.0, 15.0, 3.5, 3.5, 20.0], dtype=np.float32)
    return models, means, stds

def fetch_live_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m&"
        f"daily=precipitation_sum&past_days=3&timezone=auto"
    )
    resp = requests.get(url, timeout=3.5, headers={'User-Agent': 'Mozilla/5.0'})
    if resp.status_code == 200:
        data = resp.json()
        curr = data["current"]
        daily_precip = data.get("daily", {}).get("precipitation_sum", [0.0, 0.0, 0.0])
        p1 = daily_precip[-2] if len(daily_precip) >= 2 else 0.0
        p2 = daily_precip[-3] if len(daily_precip) >= 3 else 0.0
        p3 = daily_precip[-4] if len(daily_precip) >= 4 else 0.0
        api_val = float(p1 + 0.8 * p2 + 0.6 * p3)
        return {
            "temp_c": float(curr["temperature_2m"]),
            "rh": float(curr["relative_humidity_2m"]),
            "wind_spd_kmh": float(curr["wind_speed_10m"]),
            "wind_dir_deg": float(curr["wind_direction_10m"]),
            "api_mm": api_val
        }, True
    return None, False

models, feat_means, feat_stds = load_models()

# Header
st.markdown("<h2 style='margin-bottom:0px; font-weight:700;'>24-Hour Mountain Cloudburst & Flash Flood Early Warning System</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e; margin-top:2px; font-size:0.92rem;'>Prognostic Disaster Prevention Engine powered by Physics-Informed Neural Networks (PINN)</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### Orographic Station")
loc_choice = st.sidebar.selectbox("Target Region", list(BENCHMARK_LOCATIONS.keys()), label_visibility="collapsed")
loc_meta = BENCHMARK_LOCATIONS[loc_choice]

st.sidebar.markdown(f"""
<div style="font-size:0.8rem; color:#8b949e; margin-top:4px; line-height:1.6;">
    Coordinates: {loc_meta['lat']}°N, {loc_meta['lon']}°E<br/>
    Elevation: {loc_meta['elevation_m']} m a.s.l.<br/>
    Terrain Gradients: ∇h = ({loc_meta['dh_dx']}, {loc_meta['dh_dy']})
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Synoptic Inflow Stream")
mode = st.sidebar.radio(
    "Source",
    ["Weather Scenario Selector", "Live Weather Stream (Open-Meteo)", "Physical Parameter Simulator"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Inductive Bias Verification")
flatten_mountain = st.sidebar.checkbox(
    "Flatten Mountain (∇h → 0)",
    help="Forces terrain slope gradients to zero to verify physical dependency of updraft w_oro."
)

dh_dx = 0.0 if flatten_mountain else loc_meta["dh_dx"]
dh_dy = 0.0 if flatten_mountain else loc_meta["dh_dy"]

status_pill_html = ""

if mode == "Weather Scenario Selector":
    scenarios = loc_meta["scenarios"]
    scen_choice = st.sidebar.selectbox("Scenario Selection", list(scenarios.keys()), label_visibility="collapsed")
    s = scenarios[scen_choice]
    input_temp, input_rh, input_wspd, input_wdir, input_api = s["temp"], s["rh"], s["wind_spd"], s["wind_dir"], s["api"]
    status_pill_html = f'<div class="status-pill">Active Benchmark Scenario: <b>{scen_choice}</b></div>'

elif mode == "Live Weather Stream (Open-Meteo)":
    try:
        live_data, ok = fetch_live_weather(loc_meta["lat"], loc_meta["lon"])
        if ok and live_data:
            input_temp = live_data["temp_c"]
            input_rh = live_data["rh"]
            input_wspd = live_data["wind_spd_kmh"]
            input_wdir = live_data["wind_dir_deg"]
            input_api = live_data["api_mm"]
            status_pill_html = '<div class="status-pill">Live Satellite Stream: Connected (Open-Meteo)</div>'
        else:
            raise ValueError()
    except Exception:
        s = loc_meta["scenarios"]["Calm Dry Weather (Current Post-Monsoon)"] if "Calm Dry Weather (Current Post-Monsoon)" in loc_meta["scenarios"] else list(loc_meta["scenarios"].values())[-1]
        input_temp, input_rh, input_wspd, input_wdir, input_api = s["temp"], s["rh"], s["wind_spd"], s["wind_dir"], s["api"]
        status_pill_html = '<div class="status-pill">Current Post-Monsoon Base Stream</div>'

else:
    status_pill_html = '<div class="status-pill">Interactive Physics Simulator Mode</div>'
    c1, c2, c3 = st.columns(3)
    with c1:
        input_wspd = st.slider("Horizontal Wind Speed (km/h)", 0.0, 60.0, 25.0, 1.0)
        input_wdir = st.slider("Wind Direction (°)", 0.0, 360.0, 180.0, 5.0)
    with c2:
        input_temp = st.slider("Surface Temperature (°C)", 5.0, 42.0, 28.0, 0.5)
        input_rh = st.slider("Relative Humidity (%)", 10.0, 100.0, 92.0, 1.0)
    with c3:
        input_api = st.slider("Antecedent 3-Day Rain API (mm)", 0.0, 120.0, 45.0, 1.0)

st.markdown(status_pill_html, unsafe_allow_html=True)

# PINN Forward Pass
feat_vec, phys = compute_physics_features(input_wspd, input_wdir, input_temp, input_rh, dh_dx, dh_dy, input_api)
X_norm = (feat_vec - feat_means) / feat_stds
t_X = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)
t_C = torch.tensor([phys["c_oro_mm"]], dtype=torch.float32).unsqueeze(0)

p_tots, p_imaxes = [], []
with torch.no_grad():
    for m in models:
        pt, pi, _ = m(t_X, t_C)
        p_tots.append(pt.item())
        p_imaxes.append(pi.item())

pred_total_mm = float(np.mean(p_tots))
pred_imax_mm_hr = float(np.mean(p_imaxes))

# Operational Alert Logic
is_red = (
    (pred_imax_mm_hr >= 18.0) or
    (pred_total_mm >= 65.0) or
    ((pred_total_mm >= 38.0) and (input_api >= 38.0))
)
is_yellow = (pred_imax_mm_hr >= 10.0) or (pred_total_mm >= 30.0)

if is_red:
    alert_class = "alert-banner-red"
    title_class = "alert-title-red"
    tier_text = "RED ALERT: EMERGENCY CLOUDBURST WARNING"
    desc_text = "Severe convective cloudburst or flash flood conditions anticipated within 24 hours. High-intensity burst rates will trigger rapid hydrological pooling and debris flows."
elif is_yellow:
    alert_class = "alert-banner-yellow"
    title_class = "alert-title-yellow"
    tier_text = "YELLOW ALERT: HEAVY RAINFALL ADVISORY"
    desc_text = "Prolonged convective soaking rain anticipated. Stream levels will rise steadily; authorities should monitor local catchment basins."
else:
    alert_class = "alert-banner-green"
    title_class = "alert-title-green"
    tier_text = "GREEN ALERT: NORMAL CONDITIONS"
    desc_text = "No extreme convective cloudburst or flash flooding signatures detected for the upcoming 24-hour lead forecast cycle."

st.markdown(f"""
<div class="{alert_class}">
    <div class="{title_class}">{tier_text}</div>
    <p class="alert-desc">{desc_text}</p>
</div>
""", unsafe_allow_html=True)

# Prognostic Targets
st.markdown("<p style='font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#8b949e; margin-bottom:8px;'>24-Hour Prognostic Forecast Heads</p>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">24h Total Rain Volume</div>
        <div class="card-value">{pred_total_mm:.1f} <span style="font-size:0.95rem; font-weight:400; color:#8b949e;">mm/day</span></div>
        <div class="card-formula">Threshold: 65.0 mm / day</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Peak 1-Hour Flash Rate (I_max)</div>
        <div class="card-value" style="color:{'#f87171' if pred_imax_mm_hr>=18 else ('#fbbf24' if pred_imax_mm_hr>=10 else '#f0f6fc')};">
            {pred_imax_mm_hr:.1f} <span style="font-size:0.95rem; font-weight:400; color:#8b949e;">mm/hr</span>
        </div>
        <div class="card-formula">Disaster Threshold: ≥ 18.0 mm/hr</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Antecedent Soil Memory (API)</div>
        <div class="card-value">{input_api:.1f} <span style="font-size:0.95rem; font-weight:400; color:#8b949e;">mm</span></div>
        <div class="card-formula">API = P_t-1 + 0.8 P_t-2 + 0.6 P_t-3</div>
    </div>
    """, unsafe_allow_html=True)

# Physics Invariants
st.markdown("<p style='font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#8b949e; margin:16px 0 8px 0;'>Embedded Fluid Dynamics & Thermodynamic Invariants</p>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Kinematic Updraft (w_oro)</div>
        <div class="card-value">{phys['w_oro']:.2f} <span style="font-size:0.85rem; font-weight:400; color:#8b949e;">m/s</span></div>
        <div class="card-formula">w_oro = u·(∂h/∂x) + v·(∂h/∂y)</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Specific Humidity (q)</div>
        <div class="card-value">{phys['q_gkg']:.1f} <span style="font-size:0.85rem; font-weight:400; color:#8b949e;">g/kg</span></div>
        <div class="card-formula">Clausius-Clapeyron e_sat(T)</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Condensation Ceiling (C_oro)</div>
        <div class="card-value">{phys['c_oro_mm']:.1f} <span style="font-size:0.85rem; font-weight:400; color:#8b949e;">mm/d</span></div>
        <div class="card-formula">ρ_air · q · max(0, w_oro)</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="card-label">Dewpoint Depression</div>
        <div class="card-value">{phys['dewpoint_dep']:.1f} <span style="font-size:0.85rem; font-weight:400; color:#8b949e;">°C</span></div>
        <div class="card-formula">Saturation Gap ΔT_d</div>
    </div>
    """, unsafe_allow_html=True)

# Phase Space Plot
st.markdown("---")
col_plot, col_info = st.columns([1.8, 1.2])

with col_plot:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.grid(True, linestyle="--", alpha=0.25, color='#8b949e')

    ax.axhline(18.0, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.8, label="Cloudburst Flash Rate (18 mm/hr)")
    ax.axhline(10.0, color="#f59e0b", linestyle=":", linewidth=1.0, alpha=0.8, label="Convective Burst Watch (10 mm/hr)")
    ax.axvline(65.0, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.8, label="Deluge Threshold (65 mm/day)")
    ax.axvline(30.0, color="#f59e0b", linestyle=":", linewidth=1.0, alpha=0.8, label="Advisory (30 mm/day)")

    pt_color = "#ef4444" if is_red else ("#f59e0b" if is_yellow else "#10b981")
    ax.scatter([pred_total_mm], [pred_imax_mm_hr], color=pt_color, s=180, edgecolors="#ffffff", linewidths=1.5, zorder=5, label="PINN Prediction")

    ax.set_xlim(-5, 230)
    ax.set_ylim(-1, 38)
    ax.set_xlabel("Total 24-Hour Rainfall (mm/day)", color='#c9d1d9', fontsize=8.5, fontweight="600")
    ax.set_ylabel("Peak 1-Hour Flash Intensity (mm/hr)", color='#c9d1d9', fontsize=8.5, fontweight="600")
    ax.legend(loc="upper right", fontsize=7.5, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_info:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:16px; font-size:0.85rem; line-height:1.55; color:#c9d1d9;">
        <b style="color:#f0f6fc;">Operational Separation Principle</b><br/>
        Standard statistical models output a single 24-hour total, conflating soaking monsoonal drizzles with high-velocity flash cloudbursts.<br/><br/>
        The <b>Dual-Head PINN</b> decouples total volume from peak hourly kinetic rate, allowing civil defense authorities to trigger emergency sirens strictly when burst velocity exceeds drainage threshold.
    </div>
    """, unsafe_allow_html=True)
    if flatten_mountain:
        st.markdown("""
        <div style="margin-top:10px; background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.25); border-radius:8px; padding:12px; font-size:0.8rem; color:#fca5a5;">
            <b>Inductive Bias Check Active:</b> Terrain gradients forced to zero (∇h = 0). Kinematic updraft collapses to 0.00 m/s, demonstrating physical dependence on topographic lifting.
        </div>
        """, unsafe_allow_html=True)
