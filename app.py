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
    
    # Precise high-frequency solar position for current minute
    solpos = solarposition.get_solarposition(ts_naive, lat, lon)
    zenith = solpos['zenith'].iloc[0]
    azimuth = solpos['azimuth'].iloc[0]
    
    idx = df.index.get_indexer([ts_naive], method='nearest')[0]
    row = df.iloc[idx]
    
    opt_tilt = np.clip(zenith * 0.85, 0, 50) if zenith < 85 else 13.0
    poa_opt = irradiance.get_total_irradiance(opt_tilt, azimuth, row['dni'], row['ghi'], row
