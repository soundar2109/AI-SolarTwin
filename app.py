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
# DATA FETCHING (Open-Meteo API for VIT Chennai - 15 Day History)
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_open_meteo_live(lat=12.8406, lon=80.1534):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"],
        "past_days": 15,
        "forecast_days": 1,
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
    PUMP_POWER_W = 10.0  
    ACTUATOR_POWER_PER_DEG = 0.5  
    
    prev_tilt = 13.0
    
    for ts, row in df.iterrows():
        solpos = solarposition.get_solarposition(ts, lat, lon)
        zenith = solpos['zenith'].iloc[0]
        azimuth = solpos['azimuth'].iloc[0]
        
        # Baseline (Fixed 13° Tilt, South-facing 180° Azimuth, Uncooled)
        poa_base = irradiance.get_total_irradiance(13, 180, row['dni'], row['ghi'], row['dhi'], zenith, azimuth)['poa_global']
        t_cell_base = row['temp_air'] + (poa_base / (25.0 + 1.2 * row['wind_speed']))
        p_base = simulate_pv_power(poa_base, t_cell_base, False)
        
        # AI Digital Twin Optimization
        opt_tilt = np.clip(zenith * 0.85, 0, 50) if zenith < 85 else 13.0
        tilt_delta = abs(opt_tilt - prev_tilt)
        tilt_energy_w = tilt_delta * ACTUATOR_POWER_PER_DEG
        
        poa_opt = irradiance.get_total_irradiance(opt_tilt, azimuth, row['dni'], row['ghi'], row['dhi'], zenith, azimuth)['poa_global']
        t_cell_opt = row['temp_air'] + (poa_opt / (25.0 + 1.2 * row['wind_speed']))
        
        p_unc = simulate_pv_power(poa_opt, t_cell_opt, False)
        p_cool = simulate_pv_power(poa_opt, t_cell_opt, True) * 1.15
        
        if (p_cool - p_unc) > PUMP_POWER_W and t_cell_opt > 38.0:
            cooling_on = True
            pump_energy_w = PUMP_POWER_W
            p_gross = p_cool
        else:
            cooling_on = False
            pump_energy_w = 0.0
            p_gross = p_unc * 1.10
            
        total_parasitic_w = pump_energy_w + tilt_energy_w
        p_net = np.maximum(p_gross - total_parasitic_w, 0.0)
        
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
            'parasitic_total_w': total_parasitic_w,
            'p_base_W': p_base,
            'p_net_W': p_net,
            'zenith': zenith,
            'azimuth': azimuth
        })
        
        prev_tilt = opt_tilt
        
    return pd.DataFrame(results).set_index('timestamp')

def compute_instantaneous_state(ts_now, df, lat=12.8406, lon=80.1534):
    ts_naive = ts_now.tz_localize(None) if getattr(ts_now, 'tzinfo', None) is not None else ts_now
    solpos = solarposition.get_solarposition(ts_naive, lat, lon)
    zenith = solpos['zenith'].iloc[0]
    azimuth = solpos['azimuth'].iloc[0]
    
    idx = df.index.get_indexer([ts_naive], method='nearest')[0]
    row = df.iloc[idx]
    
    opt_tilt = np.clip(zenith * 0.85, 0, 50) if zenith < 85 else 13.0
    poa_opt = irradiance.get_total_irradiance(opt_tilt, azimuth, row['dni'], row['ghi'], row['dhi'], zenith, azimuth)['poa_global']
    t_cell_opt = row['temp_air'] + (poa_opt / (25.0 + 1.2 * row['wind_speed']))
    
    p_unc = simulate_pv_power(poa_opt, t_cell_opt, False)
    p_cool = simulate_pv_power(poa_opt, t_cell_opt, True) * 1.15
    cooling_on = True if ((p_cool - p_unc) > 10.0 and t_cell_opt > 38.0) else False
    
    return {
        'timestamp': ts_naive,
        'zenith': zenith,
        'azimuth': azimuth,
        'ghi': row['ghi'],
        'temp_air': row['temp_air'],
        'wind_speed': row['wind_speed'],
        't_cell_predicted': t_cell_opt,
        'optimal_tilt': opt_tilt,
        'cooling_active': cooling_on
    }

# ---------------------------------------------------------
# UI STREAMLIT RENDERER
# ---------------------------------------------------------
st.title("☀️ AI SolarTwin: Sensorless Digital Twin Dashboard")
st.caption("Real-Time High-Frequency Solar Tracking & Liquid Cooling Optimization | VIT Chennai (12.8406° N, 80.1534° E)")

df_raw = fetch_open_meteo_live()
df_twin = run_digital_twin_engine(df_raw)

now_ist = pd.Timestamp.now(tz='Asia/Kolkata')
live_state = compute_instantaneous_state(now_ist, df_raw)
live_time_str = now_ist.strftime('%B %d, %Y - %I:%M %p IST')

# Live Time Badge
st.info(f"🕒 **Current Simulation Timestamp:** {live_time_str} | **Sun Zenith:** {live_state['zenith']:.1f}° | **Sun Azimuth:** {live_state['azimuth']:.1f}°")

st.markdown("---")
st.subheader("🔴 Live Meteorological & Physics Model Readings")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Live GHI (Irradiance)", f"{live_state['ghi']:.1f} W/m²")
c2.metric("Ambient Temp", f"{live_state['temp_air']:.1f} °C")
c3.metric("Predicted Cell Temp", f"{live_state['t_cell_predicted']:.1f} °C")
c4.metric("AI Optimal Tilt", f"{live_state['optimal_tilt']:.1f}°")
c5.metric("Cooling Pump", "🟢 ACTIVE" if live_state['cooling_active'] else "🔴 IDLE")

# ---------------------------------------------------------
# HISTORY LOG BOOK & DATE RANGE SLICER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📖 15-Day Digital Twin History Log Book & Parasitic Audit")
st.caption("Filter historical logs by date range, audit auxiliary energy consumption, and export reports.")

col_date1, col_date2 = st.columns(2)
min_date = df_twin.index.min().date()
max_date = df_twin.index.max().date()

with col_date1:
    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
with col_date2:
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

mask = (df_twin.index.date >= start_date) & (df_twin.index.date <= end_date)
df_filtered = df_twin.loc[mask]

# ---------------------------------------------------------
# MAPPED CUMULATIVE ENERGY & EFFICIENCY METRICS (Driven by Slicer)
# ---------------------------------------------------------
total_energy_base_kWh = df_filtered['p_base_W'].sum() / 1000.0
total_energy_ai_kWh = df_filtered['p_net_W'].sum() / 1000.0
total_pump_kWh = df_filtered['pump_power_w'].sum() / 1000.0
total_tilt_kWh = df_filtered['tilt_power_w'].sum() / 1000.0
total_parasitic_kWh = total_pump_kWh + total_tilt_kWh

energy_saved_kWh = total_energy_ai_kWh - total_energy_base_kWh
efficiency_gain_pct = ((total_energy_ai_kWh - total_energy_base_kWh) / total_energy_base_kWh) * 100.0 if total_energy_base_kWh > 0 else 0.0

st.markdown("---")
st.subheader("⚡ Cumulative Energy & Parasitic Audit Summary (Selected Period)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Fixed Baseline Output", f"{total_energy_base_kWh:.2f} kWh")
m2.metric("AI Twin Net Output", f"{total_energy_ai_kWh:.2f} kWh")
m3.metric("Net Energy Saved", f"+{energy_saved_kWh:.2f} kWh", delta=f"+{energy_saved_kWh:.2f} kWh")
m4.metric("AI Efficiency Boost", f"+{efficiency_gain_pct:.1f}%", delta=f"+{efficiency_gain_pct:.1f}%")

p1, p2, p3 = st.columns(3)
p1.metric("💧 Cooling Pump Energy", f"{total_pump_kWh:.2f} kWh")
p2.metric("⚙️ Tilt Actuation Energy", f"{total_tilt_kWh:.2f} kWh")
p3.metric("🔋 Total Parasitic Overhead", f"{total_parasitic_kWh:.2f} kWh")

# ---------------------------------------------------------
# 2D PHYSICAL PANEL & FLUID DYNAMICS VISUALIZER (PIVOT ANCHORED AT CENTER)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🖥️ Physical Panel Twin & Fluid Dynamics")

tilt_deg = live_state['optimal_tilt']
tilt_rad = np.radians(tilt_deg)
panel_len = 4.0
pivot_y = 1.5

# Panel midpoint anchored directly at (0, pivot_y) on top of the mounting stand
dx = (panel_len / 2.0) * np.cos(tilt_rad)
dy = (panel_len / 2.0) * np.sin(tilt_rad)

px = [-dx, dx]
py = [pivot_y - dy, pivot_y + dy]

sun_zenith = np.clip(live_state['zenith'], 0, 90)
sun_angle_rad = np.radians(90 - sun_zenith)
sun_x = 5 * np.cos(sun_angle_rad)
sun_y = 5 * np.sin(sun_angle_rad) + 0.5

fig_twin = go.Figure()
arc_theta = np.linspace(0, np.pi, 50)
fig_twin.add_trace(go.Scatter(x=5*np.cos(arc_theta), y=5*np.sin(arc_theta)+0.5, mode='lines', line=dict(color='gray', dash='dot', width=1), showlegend=False))
fig_twin.add_trace(go.Scatter(x=[sun_x, 0], y=[sun_y, pivot_y], mode='lines', line=dict(color='orange', dash='dash', width=2), name="Solar Rays"))
fig_twin.add_trace(go.Scatter(x=[sun_x], y=[sun_y], mode='markers+text', marker=dict(size=28, color='gold'), text=["☀️ Sun"], textposition="top center", name="Sun Position"))
fig_twin.add_trace(go.Scatter(x=px, y=py, mode='lines+markers', line=dict(color='#00bc8c', width=10), name=f"Solar Panel (Tilt: {tilt_deg:.1f}°)"))
fig_twin.add_trace(go.Scatter(x=[-5, 5], y=[-0.2, -0.2], mode='lines', line=dict(color='#7f8c8d', width=3), showlegend=False))
fig_twin.add_trace(go.Scatter(x=[0, 0], y=[-0.2, pivot_y], mode='lines', line=dict(color='#7f8c8d', width=5), showlegend=False))

if live_state['cooling_active']:
    drop_x = np.linspace(px[0], px[1], 10)
    drop_y = np.linspace(py[0], py[1], 10) + 0.3
    fig_twin.add_trace(go.Scatter(x=drop_x, y=drop_y, mode='markers', marker=dict(size=8, color='#3498db', symbol='droplet'), name="Liquid Cooling Spray"))

fig_twin.update_layout(
    xaxis=dict(range=[-6, 6], visible=False),
    yaxis=dict(range=[-1, 7], visible=False),
    height=380,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_twin, use_container_width=True)

# ---------------------------------------------------------
# TIME SERIES ANALYTICS & CHARTS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 Time Series Analytics & Optimization Curves")

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=('Power Output Generation: Baseline vs AI Twin (Watts)', 'Predicted Panel Cell Temperature (°C)'),
    vertical_spacing=0.12
)

fig.add_trace(go.Scatter(x=df_twin.index, y=df_twin['p_base_W'], name='Fixed Baseline (13°)', line=dict(color='gray', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_twin.index, y=df_twin['p_net_W'], name='AI Net Power Output', line=dict(color='#00bc8c', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=df_twin.index, y=df_twin['t_cell_predicted'], name='Predicted Cell Temp', line=dict(color='#e74c3c')), row=2, col=1)
fig.add_trace(go.Scatter(x=df_twin.index, y=df_twin['temp_air'], name='Ambient Temp', line=dict(color='orange', dash='dot')), row=2, col=1)

current_ts = df_twin.index.get_indexer([now_ist.tz_localize(None)], method='nearest')[0]
closest_ts = df_twin.index[current_ts]
fig.add_vline(x=closest_ts, line_width=2, line_dash="solid", line_color="blue")
fig.add_annotation(x=closest_ts, y=1.0, yref="paper", text="Current Time", showarrow=True, arrowhead=2, arrowcolor="blue")

fig.update_layout(height=520, margin=dict(t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# HISTORY LOG TABLE & CSV DOWNLOAD
# ---------------------------------------------------------
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

csv_data = log_df.to_csv().encode('utf-8')
st.download_button(
    label="📥 Download Filtered Log Book (CSV)",
    data=csv_data,
    file_name=f"ai_solartwin_log_{start_date}_to_{end_date}.csv",
    mime="text/csv"
)
