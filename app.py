import streamlit as st
import pandas as pd
import numpy as np
import requests
import pvlib
from pvlib import solarposition, irradiance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Page Configuration
st.set_page_config(
    page_title="AI SolarTwin | Sensorless Digital Twin v2.4",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme aesthetic styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #2e3440;
        text-align: center;
    }
    .status-active {
        color: #00e676;
        font-weight: bold;
    }
    .status-idle {
        color: #ff5252;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOCATION DATA & API FETCH
# ---------------------------------------------------------
LOCATIONS = {
    "VIT Chennai (12.8406° N, 80.1534° E)": {"lat": 12.8406, "lon": 80.1534},
    "VIT Vellore (12.9692° N, 79.1559° E)": {"lat": 12.9692, "lon": 79.1559}
}

@st.cache_data(ttl=600)
def fetch_open_meteo_live(lat, lon):
    """Fetches live 24h & forecast solar meteorological data from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"],
        "past_days": 1,
        "forecast_days": 2,
        "timezone": "Asia/Kolkata"
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
        st.error(f"Error fetching live weather data: {e}")
        # Generate synthetic fallback data
        times = pd.date_range(end=pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None), periods=48, freq='h')
        return pd.DataFrame({
            'temp_air': np.random.uniform(28, 38, 48),
            'wind_speed': np.random.uniform(1.5, 4.5, 48),
            'ghi': np.maximum(0, 800 * np.sin(np.linspace(0, 4*np.pi, 48))),
            'dni': np.maximum(0, 900 * np.sin(np.linspace(0, 4*np.pi, 48))),
            'dhi': np.maximum(0, 200 * np.sin(np.linspace(0, 4*np.pi, 48)))
        }, index=times)

# ---------------------------------------------------------
# PHYSICS & DIGITAL TWIN ENGINE
# ---------------------------------------------------------
def simulate_pv_power(poa, t_cell, is_cooled):
    """Faiman model thermal prediction and temperature coefficient power loss."""
    t_effective = np.maximum(t_cell - 18.0, 25.0) if is_cooled else t_cell
    temp_loss = 1 - 0.004 * (t_effective - 25.0)
    p_out = 400.0 * (poa / 1000.0) * temp_loss
    return np.maximum(p_out, 0.0)

def run_digital_twin_engine(df, lat, lon):
    """Calculates baseline vs AI dynamic optimization for tilt and cooling."""
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
            'zenith': zenith,
            'azimuth': azimuth,
            't_cell_predicted': t_cell_opt,
            'optimal_tilt': opt_tilt,
            'cooling_active': cooling_on,
            'p_base_W': p_base,
            'p_net_W': p_net
        })
        
    return pd.DataFrame(results).set_index('timestamp')

# ---------------------------------------------------------
# 2D PHYSICS SIMULATION VISUALIZER (PLOTLY SHAPES)
# ---------------------------------------------------------
def render_2d_panel_simulation(zenith, optimal_tilt, t_cell, cooling_active, ghi):
    """Draws dynamic 2D visual representation of the solar panel, sun path, and liquid cooling."""
    fig = go.Figure()
    
    # Sky Background & Sun Trajectory Arc
    arc_x = [100 * math.cos(rad) for rad in np.linspace(0, math.pi, 100)]
    arc_y = [100 * math.sin(rad) for rad in np.linspace(0, math.pi, 100)]
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines', line=dict(color='#2c3e50', width=2, dash='dash'), showlegend=False))
    
    # Sun Angle Calculation (Zenith to 2D polar plot)
    sun_angle_deg = 180 - np.clip(zenith, 5, 175)
    sun_rad = math.radians(sun_angle_deg)
    sun_x = 90 * math.cos(sun_rad)
    sun_y = 90 * math.sin(sun_rad)
    
    # Render Sun Glow & Center Body
    if ghi > 5:
        fig.add_trace(go.Scatter(
            x=[sun_x], y=[sun_y],
            mode='markers',
            marker=dict(size=40, color='gold', opacity=0.8, line=dict(color='#ffea00', width=4)),
            name='Sun Position',
            hoverinfo='text',
            text=[f"Sun Zenith Angle: {zenith:.1f}°"]
        ))
    else:
        # Night mode
        fig.add_trace(go.Scatter(
            x=[0], y=[90],
            mode='markers',
            marker=dict(size=25, color='#7f8c8d', opacity=0.5),
            name='Night (Sun Below Horizon)',
            hoverinfo='text',
            text=["Night Time"]
        ))

    # Solar Panel Geometry & Rotation Logic
    panel_tilt = optimal_tilt if ghi > 5 else 13.0
    panel_rad = math.radians(panel_tilt)
    half_len = 35.0
    
    # Center mounting pivot point
    px0, py0 = 0.0, 15.0 
    
    # Calculate end coordinates for tilted panel board
    dx = half_len * math.cos(panel_rad)
    dy = half_len * math.sin(panel_rad)
    
    p1_x, p1_y = px0 - dx, py0 - dy
    p2_x, p2_y = px0 + dx, py0 + dy
    
    # Determine Heat Gradient Color based on Cell Temp
    if cooling_active:
        panel_color = '#00e676' # Cool Active Cyan/Green
    elif t_cell > 55:
        panel_color = '#ff1744' # Critical Hot Red
    elif t_cell > 42:
        panel_color = '#ff9100' # Warm Amber
    else:
        panel_color = '#29b6f6' # Normal Blue
        
    # Draw Panel Surface Line
    fig.add_trace(go.Scatter(
        x=[p1_x, p2_x], y=[p1_y, p2_y],
        mode='lines+markers',
        line=dict(color=panel_color, width=12),
        marker=dict(size=8, color='#ffffff'),
        name='PV Panel Array',
        hoverinfo='text',
        text=[f"Tilt: {panel_tilt:.1f}° | Cell Temp: {t_cell:.1f}°C"]
    ))
    
    # Draw Panel Stand Base Structure
    fig.add_trace(go.Scatter(
        x=[0, -10, 10, 0], y=[15, 0, 0, 15],
        mode='lines',
        fill='toself',
        fillcolor='#37474f',
        line=dict(color='#263238', width=2),
        showlegend=False
    ))
    
    # Draw Incident Sun Rays
    if ghi > 5:
        fig.add_trace(go.Scatter(
            x=[sun_x, px0], y=[sun_y, py0],
            mode='lines',
            line=dict(color='#ffd54f', width=2, dash='dot'),
            name='Solar Radiation Rays'
        ))
        
    # Active Cooling Liquid Spray Visualizer
    if cooling_active:
        # Generate fluid spray particles along panel surface
        spray_x = np.linspace(p1_x, p2_x, 15)
        spray_y = np.linspace(p1_y, p2_y, 15) + np.random.uniform(2, 5, 15)
        fig.add_trace(go.Scatter(
            x=spray_x, y=spray_y,
            mode='markers',
            marker=dict(size=8, color='#00e5ff', symbol='droplet', opacity=0.8),
            name='Liquid Cooling Spray'
        ))

    # Layout styling for clean dark component integration
    fig.update_layout(
        title=f"<b>Physical Panel Twin & Dynamic Solar Tracking</b> (Tilt: {panel_tilt:.1f}° | Cell Temp: {t_cell:.1f}°C)",
        xaxis=dict(range=[-110, 110], visible=False),
        yaxis=dict(range=[-10, 115], visible=False),
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(14,17,23,0.8)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ---------------------------------------------------------
# UI STREAMLIT APPLICATION
# ---------------------------------------------------------
# Top Bar Header
st.title("☀️ AI SolarTwin: Sensorless Digital Twin v2.4")
st.caption("Physics-Informed Real-Time Solar Tilt & Liquid-Cooling Dynamic Optimization Engine")

# Controls Header Bar
col_loc, col_status = st.columns([2, 1])
with col_loc:
    selected_loc_name = st.selectbox("📍 Select Target Location:", list(LOCATIONS.keys()))
    loc = LOCATIONS[selected_loc_name]

# Run Simulation Engine
df_raw = fetch_open_meteo_live(loc['lat'], loc['lon'])
df_twin = run_digital_twin_engine(df_raw, loc['lat'], loc['lon'])

# Extract Live Current Hour State
now_ist = pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)
closest_idx = df_twin.index.get_indexer([now_ist], method='nearest')[0]
live_state = df_twin.iloc[closest_idx]

# Display Live Digital Twin Metrics Card
st.markdown("---")
st.subheader("🔴 Live Physics & Sensorless Model State")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Live Irradiance (GHI)", f"{live_state['ghi']:.1f} W/m²")
m2.metric("Ambient Temp", f"{live_state['temp_air']:.1f} °C")
m3.metric("Predicted Cell Temp", f"{live_state['t_cell_predicted']:.1f} °C")
m4.metric("AI Optimal Tilt", f"{live_state['optimal_tilt']:.1f}°")
m5.metric("Cooling Pump Status", "🟢 ACTIVE" if live_state['cooling_active'] else "🔴 IDLE")

# Render Interactive 2D Physical Simulation
st.markdown("---")
fig_2d = render_2d_panel_simulation(
    zenith=live_state['zenith'],
    optimal_tilt=live_state['optimal_tilt'],
    t_cell=live_state['t_cell_predicted'],
    cooling_active=live_state['cooling_active'],
    ghi=live_state['ghi']
)
st.plotly_chart(fig_2d, use_container_width=True)

# Analytics Curves & Performance Graphs
st.markdown("---")
st.subheader("📈 Time-Series Optimization Curves")

fig_ts = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=('Power Generation Performance: Baseline (13° Fixed) vs AI Twin (Watts)', 'Panel Cell Temperature (°C) & Active Thermal Management'),
    vertical_spacing=0.12
)

# Power Curve
fig_ts.add_trace(go.Scatter(x=df_twin.index, y=df_twin['p_base_W'], name='Fixed Baseline (13°)', line=dict(color='gray', dash='dash')), row=1, col=1)
fig_ts.add_trace(go.Scatter(x=df_twin.index, y=df_twin['p_net_W'], name='AI Net Power Output', line=dict(color='#00e676', width=2.5)), row=1, col=1)

# Temperature Curve
fig_ts.add_trace(go.Scatter(x=df_twin.index, y=df_twin['t_cell_predicted'], name='Predicted Cell Temp', line=dict(color='#ff1744')), row=2, col=1)
fig_ts.add_trace(go.Scatter(x=df_twin.index, y=df_twin['temp_air'], name='Ambient Air Temp', line=dict(color='#ff9100', dash='dot')), row=2, col=1)

# Current Time Highlight Line
current_ts = df_twin.index[closest_idx]
fig_ts.add_vline(x=current_ts, line_width=2, line_dash="solid", line_color="#29b6f6")
fig_ts.add_annotation(x=current_ts, y=1.0, yref="paper", text="Current Live Hour", showarrow=True, arrowhead=2, arrowcolor="#29b6f6")

fig_ts.update_layout(height=520, margin=dict(t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,0.8)')
st.plotly_chart(fig_ts, use_container_width=True)
