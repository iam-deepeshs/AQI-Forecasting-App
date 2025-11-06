import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None
from sklearn.preprocessing import StandardScaler

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title="🌏 AQI Forecast Dashboard", layout="wide")

# ------------------------- HEADER -------------------------
st.title("🌏 India Air Quality Forecast Dashboard")
st.caption("Predicting and visualizing Air Quality Index (AQI) using LSTM & GRU models")

st.sidebar.header("📂 Upload or Use Default Data")

# ------------------------- DATA UPLOAD -------------------------
uploaded_file = st.sidebar.file_uploader("Upload your AQI CSV (city_day.csv or AQI_India.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Use default file in data folder
    try:
        df = pd.read_csv("data/city_day_small.csv")
        st.sidebar.success("✅ Using default AQI_India.csv from /data folder")
    except FileNotFoundError:
        st.error("❌ No file uploaded and default dataset not found.")
        st.stop()

# ------------------------- DATA PREPROCESS -------------------------
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['city', 'aqi'])

# City selector
city_list = sorted(df['city'].dropna().unique())
city = st.sidebar.selectbox("🏙️ Select a City", city_list)

# Subset data
df_city = df[df['city'] == city].sort_values('date')
st.subheader(f"📊 AQI Trend for {city}")
st.line_chart(df_city.set_index('date')['aqi'])

# ------------------------- MODEL CHOICE -------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio("Choose Model", ["LSTM", "GRU"])
model_path = f"models/{model_choice.lower()}_model.h5"

try:
    model = load_model(model_path)
    st.sidebar.success(f"✅ {model_choice} model loaded successfully!")
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not load {model_choice} model: {e}")
    st.stop()

# ------------------------- FEATURE SCALING -------------------------
features = [f for f in ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3'] if f in df_city.columns]

if len(features) == 0:
    st.warning("No pollutant feature columns found in your data. Showing only AQI forecast demo.")
    df_city['pm2.5'] = df_city['aqi']

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(df_city[features])
y = scaler_y.fit_transform(df_city[['aqi']])

# Make sequences for time series
def make_sequences(X, y, lookback=30):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

LOOKBACK = 30
X_seq, y_seq = make_sequences(X, y)

# ------------------------- PREDICTION -------------------------
y_pred_scaled = model.predict(X_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
df_pred = df_city.iloc[LOOKBACK:].copy()
df_pred['Predicted AQI'] = y_pred

# ------------------------- CHART -------------------------
st.subheader(f"📈 {model_choice} Model — Actual vs Predicted AQI ({city})")

fig = px.line(df_pred, x='date', y=['aqi', 'Predicted AQI'],
              labels={'value': 'AQI', 'variable': 'Type'},
              title=f"{city} AQI — Actual vs Predicted ({model_choice})")
st.plotly_chart(fig, use_container_width=True)

# ------------------------- 7-DAY FORECAST -------------------------
st.subheader("🔮 7-Day AQI Forecast")

last_window = X[-LOOKBACK:, :]
preds_scaled = []
window = last_window.copy()

for _ in range(7):
    pred = model.predict(window[np.newaxis, :, :])[0][0]
    preds_scaled.append(pred)
    window = np.roll(window, -1, axis=0)
    window[-1, :] = pred

future_aqi = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
future_dates = pd.date_range(df_city['date'].iloc[-1] + pd.Timedelta(days=1), periods=7)

forecast_df = pd.DataFrame({"Date": future_dates, "Predicted AQI": future_aqi})
st.dataframe(forecast_df.style.format({"Predicted AQI": "{:.2f}"}))

fig_forecast = px.line(forecast_df, x='Date', y='Predicted AQI',
                       title=f"7-Day Forecast for {city} ({model_choice} Model)")
st.plotly_chart(fig_forecast, use_container_width=True)

# ------------------------- FOOTER -------------------------
st.markdown("---")
st.caption("Developed by **Deepesh Srivastava** — Streamlit AQI Forecast App 🌿")
