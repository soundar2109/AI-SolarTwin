import streamlit as st
import pandas as pd
import numpy as np
import requests
import pvlib
from pvlib import solarposition, irradiance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="AI SolarTwin | Digital Twin Dashboard",
    page_icon="☀️",
    layout="wide"
)

# ---------------------------------------------------------
# DATA FETCHING (Open-Meteo API for VIT Chennai)
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_open_meteo_live(lat=12.8406, lon=80.1534):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"],
        "past_days": 1,
        "forecast_days": 2,
        "timezone": "Asia/Kolkata"
    }
    res = requests.get(url, params=params).json()
    hourly = res['hourly']
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time']),
        'temp_air': hourly['temperature_2m'],
        'wind_speed': hourly['wind_speed_10m'],
        'ghi': hourly['shortwave_radiation'],
        'dni': hourly['direct_normal_irradiance'],
        'dhi': hourly['diffuse_radiation']
    }).set_index('timestamp')
    return df

def simulate_pv_power(poa, t_cell, is_cooled):
    t_effective = np.maximum(t_cell - 18.0, 25.0) if is_cooled else t_cell
    temp_loss = 1 - 0.004 * (t_effective - 25.0)
    p_out = 400.0 * (poa / 1000.0) * temp_loss
    return np.maximum(p_out, 0.0)

def run_digital_twin_engine(df, lat=12.8406, lon=80.1534):
    results = []
    PUMP_POWER_W = 15.0
    
    for ts, row in df.iterrows():
        solpos = solarposition.get_solarposition(ts, lat, lon)
        zenith = solpos['zenith'].iloc[0]
        azimuth = solpos['azimuth'].iloc[0]
        
        # Baseline (Fixed 13° Tilt, Uncooled)
        poa_base = irradiance.get_total_irradiance(13, 180, row['dni'], row['ghi'], row['dhi'], zenith, azimuth)['poa_global']
        t_cell_base = row['temp_air'] + (poa_base / (25.0 + 1.2 * row['wind_speed']))
        p_base = simulate_pv_power(poa_base, t_cell_base, False)
        
        # AI Digital Twin Optimization
        opt_tilt = np.clip(zenith, 0, 60) if zenith < 85 else 13.0
        poa_opt = irradiance.get_total_irradiance(opt_tilt, azimuth, row['dni'], row['ghi'], row['dhi'], zenith, azimuth)['poa_global']
        t_cell_opt = row['temp_air'] + (poa_opt / (25.0 + 1.2 * row['wind_speed']))
        
        p_unc = simulate_pv_power(poa_opt, t_cell_opt, False)
        p_cool = simulate_pv_power(poa_opt, t_cell_opt, True)
        
        if (p_cool - p_unc) > PUMP_POWER_W and t_cell_opt > 42.0:
            cooling_on = True
            p_net = p_cool - PUMP_POWER_W
        else:
            cooling_on = False
            p_net = p_unc
            
        results.append({
            'timestamp': ts,
            'ghi': row['ghi'],
            'temp_air': row['temp_air'],
            'wind_speed': row['wind_speed'],
            't_cell_predicted': t_cell_opt,
            'optimal_tilt': opt_tilt,
            'cooling_active': cooling_on,
            'p_base_W': p_base,
            'p_net_W': p_net
        })
        
    return pd.DataFrame(results).set_index('timestamp')

# ---------------------------------------------------------
# UI STREAMLIT RENDERER
# ---------------------------------------------------------
st.title("☀️ AI SolarTwin: Sensorless Digital Twin Dashboard")
st.caption("Live Sensorless Solar Optimization Engine | Coordinates: VIT Chennai (12.8406° N, 80.1534° E)")

df_raw = fetch_open_meteo_live()
df_twin = run_digital_twin_engine(df_raw)

# Get current hour reading
now_ist = pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)
closest_idx = df_twin.index.get_indexer([now_ist], method='nearest')[0]
live_state = df_twin.iloc[closest_idx]

st.markdown("---")
st.subheader("🔴 Live Meteorological & Physics Model Readings")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Live GHI (Irradiance)", f"{live_state['ghi']:.1f} W/m²")
c2.metric("Ambient Temp", f"{live_state['temp_air']:.1f} °C")
c3.metric("Predicted Cell Temp", f"{live_state['t_cell_predicted']:.1f} °C")
c4.metric("AI Optimal Tilt", f"{live_state['optimal_tilt']:.1f}°")
c5.metric("Cooling Pump", "🟢 ACTIVE" if live_state['cooling_active'] else "🔴 IDLE")

# Analytics Section
st.markdown("---")
st.subheader("📈 Time Series Analytics & Optimization Curve")

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=('Power Output Generation: Baseline vs AI Twin (Watts)', 'Predicted Panel Cell Temperature (°C)'),
    vertical_spacing=0.12
)

fig.add_trace(go.Scatter(x=df_twin.index, y=df_twin['p_base_W'], name='Fixed Baseline (13°)', line=dict(color='gray', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_twin.index, y=df_twin['p_net_W'], name='AI Net Power Output', line=dict(color='#00bc8c', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=df_twin.index, y=df_twin['t_cell_predicted'], name='Predicted Cell Temp', line=dict(color='#e74c3c')), row=2, col=1)
fig.add_trace(go.Scatter(x=df_twin.index, y=df_twin['temp_air'], name='Ambient Temp', line=dict(color='orange', dash='dot')), row=2, col=1)

current_ts = df_twin.index[closest_idx]
fig.add_vline(x=current_ts, line_width=2, line_dash="solid", line_color="blue")
fig.add_annotation(x=current_ts, y=1.0, yref="paper", text="Current Live Hour", showarrow=True, arrowhead=2, arrowcolor="blue")

fig.update_layout(height=520, margin=dict(t=40, b=20))
st.plotly_chart(fig, use_container_width=True)
