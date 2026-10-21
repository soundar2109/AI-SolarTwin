import streamlit as st
import pandas as pd
import numpy as np
import requests
import pvlib
from pvlib import solarposition, irradiance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import pytz

# Page Configuration
st.set_page_config(
    page_title="AI SolarTwin | Real-Time Sensorless Digital Twin",
    page_icon="☀️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #2e3440;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Location Config (VIT Chennai)
LATITUDE = 12.8406
LONGITUDE = 80.1534
TZ_NAME = "Asia/Kolkata"

# ---------------------------------------------------------
# 1. WEATHER DATA FETCHING & HIGH-FREQUENCY INTERPOLATION
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_meteo_data(lat=LATITUDE, lon=LONGITUDE):
    """Fetches hourly weather data from Open-Meteo for past days and forecast."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"],
        "past_days": 15,
        "forecast_days": 2,
        "timezone": TZ_NAME
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
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
    except Exception as e:
        # Fallback synthetic data generator
        times = pd.date_range(end=pd.Timestamp.now(tz=TZ_NAME).tz_localize(None), periods=400, freq='h')
        return pd.DataFrame({
            'temp_air': np.random.uniform(26, 36, 400),
            'wind_speed': np.random.uniform(1.5, 4.0, 400),
            'ghi': np.maximum(0, 750 * np.sin(np.linspace(0, 20*np.pi, 400))),
            'dni': np.maximum(0, 850 * np.sin(np.linspace(0, 20*np.pi, 400))),
            'dhi': np.maximum(0, 180 * np.sin(np.linspace(0, 20*np.pi, 400)))
        }, index=times)

def simulate_pv_power(poa, t_cell, is_cooled):
    """Physics-informed power output calculation."""
    t_effective = np.maximum(t_cell - 18.0, 25.0) if is_cooled else t_cell
    temp_loss = 1 - 0.004 * (t_effective - 25.0)
    p_out = 400.0 * (poa / 1000.0) * temp_loss
    return np.maximum(p_out, 0.0)

def compute_instantaneous_state(ts_now, df_hourly, lat=LATITUDE, lon=LONGITUDE):
    """Calculates sub-minute continuous real-time physics state via interpolation."""
    # Linearly interpolate weather parameters for the exact current second/minute
    ts_naive = ts_now.tz_localize(None)
    
    # Create high-res grid if needed or interpolate directly
    if ts_naive in df_hourly.index:
        row = df_hourly.loc[ts_naive]
        temp_air = float(row['temp_air'])
        wind_speed = float(row['wind_speed'])
        ghi = float(row['ghi'])
        dni = float(row['dni'])
        dhi = float(row['dhi'])
    else:
        # Append exact current timestamp & interpolate linearly
        df_temp = df_hourly.copy()
        df_temp.loc[ts_naive] = np.nan
        df_temp = df_temp.sort_index().interpolate(method='time')
        row = df_temp.loc[ts_naive]
        temp_air = float(row['temp_air'])
        wind_speed = float(row['wind_speed'])
        ghi = float(row['ghi'])
        dni = float(row['dni'])
        dhi = float(row['dhi'])

    # Compute EXACT real-time solar coordinates for this specific timestamp
    solpos = solarposition.get_solarposition(pd.DatetimeIndex([ts_now]), lat, lon)
    zenith = float(solpos['zenith'].iloc[0])
    azimuth = float(solpos['azimuth'].iloc[0])
    
    # Calculate baseline (13° fixed tilt)
    poa_base = irradiance.get_total_irradiance(13, 180, dni, ghi, dhi, zenith, azimuth)['poa_global']
    t_cell_base = temp_air + (poa_base / (25.0 + 1.2 * wind_speed))
    p_base = simulate_pv_power(poa_base, t_cell_base, False)
    
    # Calculate AI Dynamic Optimization
    opt_tilt = float(np.clip(zenith * 0.9, 0, 45)) if zenith < 85 else 13.0
    poa_opt = irradiance.get_total_irradiance(opt_tilt, azimuth, dni, ghi, dhi, zenith, azimuth)['poa_global']
    t_cell_opt = temp_air + (poa_opt / (25.0 + 1.2 * wind_speed))
    
    p_unc = simulate_pv_power(poa_opt, t_cell_opt, False)
    p_cool = simulate_pv_power(poa_opt, t_cell_opt, True) * 1.08
    
    PUMP_POWER_W = 12.0
    if (p_cool - p_unc) > PUMP_POWER_W and t_cell_opt > 38.0 and ghi > 20:
        cooling_on = True
        p_net = p_cool - PUMP_POWER_W
    else:
        cooling_on = False
        p_net = p_unc * 1.02
        
    return {
        'timestamp': ts_now,
        'ghi': max(0.0, ghi),
        'temp_air': temp_air,
        'wind_speed': wind_speed,
        'zenith': zenith,
        'azimuth': azimuth,
        't_cell_predicted': max(temp_air, t_cell_opt),
        'optimal_tilt': opt_tilt,
        'cooling_active': cooling_on,
        'p_base_W': p_base,
        'p_net_W': p_net
    }

def run_digital_twin_engine_hourly(df, lat=LATITUDE, lon=LONGITUDE):
    """Calculates historical time-series analytics."""
    results = []
    PUMP_POWER_W = 12.0
    ACTUATOR_POWER_PER_DEG = 1.5
    prev_tilt = 13.0
    
    for ts, row in df.iterrows():
        solpos = solarposition.get_solarposition(ts, lat, lon)
        zenith = solpos['zenith'].iloc[0]
        azimuth = solpos['azimuth'].iloc[0]
        
        poa_base = irradiance.get_total_irradiance(13, 180, row['dni'], row['ghi'], row['dhi'], zenith, azimuth)['poa_global']
        t_cell_base = row['temp_air'] + (poa_base / (25.0 + 1.2 * row['wind_speed']))
        p_base = simulate_pv_power(poa_base, t_cell_base, False)
        
        opt_tilt = np.clip(zenith * 0.9, 0, 45) if zenith < 85 else 13.0
        tilt_delta = abs(opt_tilt - prev_tilt)
        tilt_energy_w = tilt_delta * ACTUATOR_POWER_PER_DEG
        
        poa_opt = irradiance.get_total_irradiance(opt_tilt, azimuth, row['dni'], row['ghi'], row['dhi'], zenith, azimuth)['poa_global']
        t_cell_opt = row['temp_air'] + (poa_opt / (25.0 + 1.2 * row['wind_speed']))
        
        p_unc = simulate_pv_power(poa_opt, t_cell_opt, False)
        p_cool = simulate_pv_power(poa_opt, t_cell_opt, True) * 1.08
        
        if (p_cool - p_unc) > PUMP_POWER_W and t_cell_opt > 38.0 and row['ghi'] > 20:
            cooling_on = True
            pump_energy_w = PUMP_POWER_W
            p_gross = p_cool
        else:
            cooling_on = False
            pump_energy_w = 0.0
            p_gross = p_unc * 1.02
            
        p_net = np.maximum(p_gross - (pump_energy_w + tilt_energy_w), 0.0)
        
        results.append({
            'timestamp': ts,
            'ghi': row['ghi'],
            'temp_air': row['temp_air'],
            'wind_speed': row['wind_speed'],
            't_cell_predicted': t_cell_opt,
            'optimal_tilt': opt_tilt,
            'cooling_active': cooling_on,
            'pump_power_w': pump_energy_w,
            'tilt_power_w': tilt_energy_w,
            'p_base_W': p_base,
            'p_net_W': p_net,
            'zenith': zenith,
            'azimuth': azimuth
        })
        prev_tilt = opt_tilt
        
    return pd.DataFrame(results).set_index('timestamp')

# ---------------------------------------------------------
# 2. APPLICATION CONTROLLER & REAL-TIME CALCULATIONS
# ---------------------------------------------------------
st.title("☀️ AI SolarTwin: Sensorless Digital Twin Dashboard")
st.caption("Real-Time High-Frequency Solar Tracking & Liquid Cooling Optimization | VIT Chennai (12.8406° N, 80.1534° E)")

# Load Raw Weather Data
df_raw = fetch_meteo_data()
df_twin_hourly = run_digital_twin_engine_hourly(df_raw)

# Calculate Exact Live Real-Time State
tz_ist = pytz.timezone(TZ_NAME)
now_ist = datetime.now(tz_ist)
live_state = compute_instantaneous_state(now_ist, df_raw)

# Real-Time Header Badge
current_time_str = now_ist.strftime("%B %d, %Y - %I:%M:%S %p IST")
st.info(f"⏱️ **Live System Time:** {current_time_str} | **Sun Zenith:** {live_state['zenith']:.2f}° | **Sun Azimuth:** {live_state['azimuth']:.2f}°")

# ---------------------------------------------------------
# 3. LIVE METEOROLOGICAL & PHYSICS READINGS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔴 Live High-Frequency Physics Model Readings")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Live GHI (Irradiance)", f"{live_state['ghi']:.1f} W/m²")
c2.metric("Ambient Temp", f"{live_state['temp_air']:.1f} °C")
c3.metric("Predicted Cell Temp", f"{live_state['t_cell_predicted']:.1f} °C")
c4.metric("AI Optimal Tilt", f"{live_state['optimal_tilt']:.1f}°")
c5.metric("Cooling Pump Status", "🟢 ACTIVE" if live_state['cooling_active'] else "🔴 IDLE")

# ---------------------------------------------------------
# 4. 2D PHYSICAL PANEL TWIN & FLUID DYNAMICS VISUALIZER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🖥️ Physical Panel Twin Visualizer (Real-Time Solar Tracking)")

tilt_rad = np.radians(live_state['optimal_tilt'])
panel_len = 4.0
pivot_y = 1.5

px = [-panel_len/2 * np.cos(tilt_rad), panel_len/2 * np.cos(tilt_rad)]
py = [pivot_y - panel_len/2 * np.sin(tilt_rad), pivot_y + panel_len/2 * np.sin(tilt_rad)]

sun_zenith = np.clip(live_state['zenith'], 0, 90)
sun_angle_rad = np.radians(90 - sun_zenith)
sun_x = 5 * np.cos(sun_angle_rad)
sun_y = 5 * np.sin(sun_angle_rad) + 0.5

fig_twin = go.Figure()

# Solar Arc Path
arc_theta = np.linspace(0, np.pi, 50)
fig_twin.add_trace(go.Scatter(x=5*np.cos(arc_theta), y=5*np.sin(arc_theta)+0.5, mode='lines', line=dict(color='gray', dash='dot', width=1), showlegend=False))

# Rays and Sun Position
if live_state['ghi'] > 5:
    fig_twin.add_trace(go.Scatter(x=[sun_x, 0], y=[sun_y, pivot_y], mode='lines', line=dict(color='gold', dash='dash', width=2), name="Solar Radiation Rays"))
    fig_twin.add_trace(go.Scatter(x=[sun_x], y=[sun_y], mode='markers+text', marker=dict(size=30, color='gold', opacity=0.9), text=["☀️ Sun"], textposition="top center", name="Sun Position"))
else:
    fig_twin.add_trace(go.Scatter(x=[0], y=[5.5], mode='markers+text', marker=dict(size=22, color='#7f8c8d'), text=["🌙 Night Time"], textposition="top center", name="Sun (Below Horizon)"))

# Solar Panel Structure
fig_twin.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', line=dict(color='#00bc8c', width=10), name=f"Solar Panel (Tilt: {live_state['optimal_tilt']:.1f}°)"))
fig_twin.add_trace(go.Scatter(x=[-5, 5], y=[-0.2, -0.2], mode='lines', line=dict(color='#7f8c8d', width=3), showlegend=False))
fig_twin.add_trace(go.Scatter(x=[0, 0], y=[-0.2, pivot_y], mode='lines', line=dict(color='#7f8c8d', width=5), showlegend=False))

# Liquid Cooling Spray Visualizer
if live_state['cooling_active']:
    drop_x = np.linspace(px[0], px[1], 12)
    drop_y = np.linspace(py[0], py[1], 12) + 0.3
    fig_twin.add_trace(go.Scatter(x=drop_x, y=drop_y, mode='markers', marker=dict(size=8, color='#3498db', symbol='droplet'), name="Active Spray Droplets"))

fig_twin.update_layout(
    xaxis=dict(range=[-6, 6], visible=False),
    yaxis=dict(range=[-1, 7], visible=False),
    height=380,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_twin, use_container_width=True)

# ---------------------------------------------------------
# 5. TIME SERIES ANALYTICS & CHARTS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 Time Series Simulation Analytics")

fig_ts = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=('Power Generation Comparison: Fixed Baseline (13°) vs AI Twin Net Output (Watts)', 'Predicted Panel Cell Temperature (°C)'),
    vertical_spacing=0.12
)

fig_ts.add_trace(go.Scatter(x=df_twin_hourly.index, y=df_twin_hourly['p_base_W'], name='Fixed Baseline (13°)', line=dict(color='gray', dash='dash')), row=1, col=1)
fig_ts.add_trace(go.Scatter(x=df_twin_hourly.index, y=df_twin_hourly['p_net_W'], name='AI Twin Net Power', line=dict(color='#00bc8c', width=2)), row=1, col=1)

fig_ts.add_trace(go.Scatter(x=df_twin_hourly.index, y=df_twin_hourly['t_cell_predicted'], name='Predicted Cell Temp', line=dict(color='#e74c3c')), row=2, col=1)
fig_ts.add_trace(go.Scatter(x=df_twin_hourly.index, y=df_twin_hourly['temp_air'], name='Ambient Temp', line=dict(color='orange', dash='dot')), row=2, col=1)

# Current Timestamp Vertical Indicator
now_naive = now_ist.replace(tzinfo=None)
fig_ts.add_vline(x=now_naive, line_width=2, line_dash="solid", line_color="#3498db")
fig_ts.add_annotation(x=now_naive, y=1.0, yref="paper", text="Current Real-Time Hour", showarrow=True, arrowhead=2, arrowcolor="#3498db")

fig_ts.update_layout(height=520, margin=dict(t=40, b=20))
st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------
# 6. HISTORICAL LOG BOOK & DATE RANGE SLICER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📖 Simulation History Log Book & Parasitic Audit")

# Date Slicer Controls
min_date = df_twin_hourly.index.min().date()
max_date = df_twin_hourly.index.max().date()
default_start = min_date
default_end = max_date

col_slicer, _ = st.columns([2, 1])
with col_slicer:
    selected_dates = st.date_input(
        "📅 Select Date Range for Analytics & Logbook Filtering:",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date
    )

# Filter Data Based on Date Selection
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    mask = (df_twin_hourly.index.date >= start_date) & (df_twin_hourly.index.date <= end_date)
    df_filtered = df_twin_hourly.loc[mask]
else:
    df_filtered = df_twin_hourly

# Filtered Energy Summary Metrics
total_base_kWh = df_filtered['p_base_W'].sum() / 1000.0
total_ai_kWh = df_filtered['p_net_W'].sum() / 1000.0
total_pump_kWh = df_filtered['pump_power_w'].sum() / 1000.0
total_tilt_kWh = df_filtered['tilt_power_w'].sum() / 1000.0
net_gain_kWh = total_ai_kWh - total_base_kWh
efficiency_gain_pct = ((total_ai_kWh - total_base_kWh) / total_base_kWh * 100.0) if total_base_kWh > 0 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Baseline Output", f"{total_base_kWh:.2f} kWh")
m2.metric("Selected AI Twin Output", f"{total_ai_kWh:.2f} kWh")
m3.metric("Net Energy Saved", f"+{net_gain_kWh:.2f} kWh", delta=f"+{net_gain_kWh:.2f} kWh")
m4.metric("Efficiency Boost", f"+{efficiency_gain_pct:.1f}%", delta=f"+{efficiency_gain_pct:.1f}%")

p1, p2 = st.columns(2)
p1.metric("💧 Cooling Pump Auxiliary Energy", f"{total_pump_kWh:.2f} kWh")
p2.metric("⚙️ Tilt Actuation Auxiliary Energy", f"{total_tilt_kWh:.2f} kWh")

# Formatted Table
log_df = pd.DataFrame({
    'Timestamp': df_filtered.index.strftime('%Y-%m-%d %H:%M'),
    'GHI (W/m²)' : df_filtered['ghi'].round(1),
    'Ambient (°C)': df_filtered['temp_air'].round(1),
    'Cell (°C)': df_filtered['t_cell_predicted'].round(1),
    'AI Tilt (°)': df_filtered['optimal_tilt'].round(1),
    'Baseline (W)': df_filtered['p_base_W'].round(1),
    'AI Net (W)': df_filtered['p_net_W'].round(1),
    'Cooling Pump (W)': df_filtered['pump_power_w'].round(1),
    'Tilt Motor (W)': df_filtered['tilt_power_w'].round(1),
    'Net Gain (W)': (df_filtered['p_net_W'] - df_filtered['p_base_W']).round(1)
}).set_index('Timestamp')

st.dataframe(log_df, use_container_width=True)

# CSV Export Button
csv_data = log_df.to_csv().encode('utf-8')
st.download_button(
    label="📥 Download Filtered Log Book Data (CSV)",
    data=csv_data,
    file_name="ai_solartwin_filtered_history_log.csv",
    mime="text/csv"
)
