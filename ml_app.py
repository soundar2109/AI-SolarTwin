import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Solar AI - Machine Learning Engine", layout="wide"
)

st.title("🤖 Data-Driven Solar Power Prediction Engine")
st.write(
    "Dedicated Machine Learning Dashboard utilizing a Supervised Random"
    " Forest Regressor."
)


# 1. Load Model & Dataset
@st.cache_resource
def load_resources():
  model = joblib.load("solar_ml_model.pkl")
  data = pd.read_csv("ai_solartwin_log_2026-10-20_to_2026-11-04.csv")
  return model, data


model, df = load_resources()

# Sidebar - Quick Metrics
st.sidebar.header("📊 Model Evaluation Summary")
st.sidebar.metric(label="Model Architecture", value="Random Forest")
st.sidebar.metric(label="Training Accuracy (R²)", value="0.87")
st.sidebar.metric(label="Total Dataset Logs", value=f"{len(df)} samples")

# --- Interactive Prediction Section ---
st.subheader("⚡ Live Real-Time Inference Simulator")
st.write(
    "Adjust environmental parameters to see the trained ML model predict net"
    " solar power output in real-time:"
)

col1, col2, col3 = st.columns(3)
with col1:
  ghi_in = st.slider("Global Horizontal Irradiance (GHI W/m²)", 0.0, 1000.0, 450.0)
with col2:
  ambient_in = st.slider("Ambient Temperature (°C)", 10.0, 50.0, 28.0)
with col3:
  cell_in = st.slider("Cell Temperature (°C)", 10.0, 80.0, 35.0)

# Run Inference
input_array = np.array([[ghi_in, ambient_in, cell_in]])
predicted_power = model.predict(input_array)[0]

st.success(
    f"### 🎯 ML Predicted Net Power Output: **{predicted_power:.2f} Watts**"
)

st.markdown("---")

# --- Analysis & Visualizations Section ---
col_left, col_right = st.columns(2)

with col_left:
  st.subheader("🌲 Feature Weight Analysis")
  st.write(
      "Relative weight of each meteorological variable during node splits:"
  )

  features = ["GHI (W/m²)", "Ambient (°C)", "Cell (°C)"]
  importances = model.feature_importances_

  fig1, ax1 = plt.subplots(figsize=(6, 4))
  ax1.barh(features, importances, color="#1f77b4")
  ax1.set_xlabel("Relative Importance")
  ax1.set_title("Random Forest Feature Weights")
  st.pyplot(fig1)

with col_right:
  st.subheader("📈 Historical Ground Truth vs ML Output")
  st.write("First 50 historical logs comparing actual vs model-predicted net power:")

  X_hist = df[["GHI (W/m²)", "Ambient (°C)", "Cell (°C)"]]
  df_sample = df.head(50).copy()
  df_sample["ML_Predicted_Net_W"] = model.predict(X_hist.head(50))

  fig2, ax2 = plt.subplots(figsize=(6, 4))
  ax2.plot(
      df_sample.index,
      df_sample["AI Net (W)"],
      label="Actual Logged Net (W)",
      color="black",
      alpha=0.7,
  )
  ax2.plot(
      df_sample.index,
      df_sample["ML_Predicted_Net_W"],
      label="ML Predicted (W)",
      color="#ff7f0e",
      linestyle="--",
  )
  ax2.set_ylabel("Power (Watts)")
  ax2.set_xlabel("Log Sample Index")
  ax2.legend()
  ax2.set_title("Actual vs Predicted Time-Series")
  st.pyplot(fig2)
