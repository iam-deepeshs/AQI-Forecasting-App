# app.py
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ----- Try TensorFlow (optional) -----
try:
    from tensorflow.keras.models import load_model  # type: ignore
    TF_AVAILABLE = True
except Exception:
    load_model = None  # type: ignore
    TF_AVAILABLE = False

# ================= CONFIG =================
st.set_page_config(page_title="🌏 AQI Forecast Dashboard", layout="wide")
st.title("🌏 India Air Quality Forecast Dashboard")
st.caption("Predicting and visualizing AQI with LSTM/GRU (auto-fallback if TF not available).")
st.sidebar.header("📂 Data & Model")

# ================= HELPERS =================
@st.cache_data(show_spinner=False)
def load_csv_smart(uploaded):
    """Read CSV from upload or common paths; normalize columns."""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        # Try common locations
        candidates = [
            "data/city_day_small.csv",
            "city_day_small.csv",
            "data/AQI_India.csv",
            "AQI_India.csv",
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            st.error("❌ No file uploaded and no default dataset found.")
            st.stop()
        df = pd.read_csv(path)
        st.sidebar.success(f"✅ Using default dataset: {path}")

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    # Minimal required columns
    required = {"date", "city", "aqi"}
    if not required.issubset(set(df.columns)):
        # Try to map likely names
        rename_map = {
            "aqi_bucket": "aqi_bucket",
            "pm2.5": "pm2.5", "pm10": "pm10",
            "no2": "no2", "so2": "so2", "co": "co", "o3": "o3", "nh3": "nh3",
            "date": "date", "city": "city", "aqi": "aqi"
        }
        df = df.rename(columns=rename_map)

    # Parse dates & basic cleaning
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["city", "aqi"])
    return df

def make_sequences(X, y, lookback=30):
    if len(X) <= lookback:
        return np.empty((0, lookback, X.shape[1])), np.empty((0, 1))
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

@st.cache_resource(show_spinner=False)
def load_keras_model_safe(model_choice):
    """Load Keras model if TF is available; search in root and /models."""
    if not TF_AVAILABLE:
        return None, "TensorFlow not available in this environment. Running demo mode."
    model_files = [
        f"{model_choice.lower()}_model.h5",
        os.path.join("models", f"{model_choice.lower()}_model.h5"),
    ]
    for p in model_files:
        if os.path.exists(p):
            try:
                return load_model(p), f"{model_choice} model loaded: {p}"
            except Exception as e:
                return None, f"Could not load {model_choice} model ({p}): {e}"
    return None, f"No model file found for {model_choice} in {model_files}"

# ================= DATA =================
uploaded_file = st.sidebar.file_uploader(
    "Upload AQI CSV (e.g., city_day.csv / AQI_India.csv)", type=["csv"]
)
df = load_csv_smart(uploaded_file)

# City selector
cities = sorted(df["city"].dropna().unique().tolist())
city = st.sidebar.selectbox("🏙️ Select a City", cities)

# Subset & show base trend
df_city = df[df["city"] == city].sort_values("date")
st.subheader(f"📊 AQI Trend — {city}")
st.line_chart(df_city.set_index("date")["aqi"])

# ================= MODEL =================
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio("Choose Model", ["LSTM", "GRU"])

model, model_msg = load_keras_model_safe(model_choice)
if model_msg:
    st.sidebar.info(model_msg)

# ================= FEATURES & SCALING =================
feature_candidates = ["pm2.5", "pm10", "no2", "so2", "co", "o3", "nh3"]
features = [f for f in feature_candidates if f in df_city.columns]

if len(features) == 0:
    # Fallback: derive a synthetic feature from AQI so scaler works
    df_city["pm2.5"] = df_city["aqi"]
    features = ["pm2.5"]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(df_city[features].values.astype(float))
y = scaler_y.fit_transform(df_city[["aqi"]].values.astype(float))

LOOKBACK = 30
X_seq, y_seq = make_sequences(X, y, lookback=LOOKBACK)

# ================= PREDICTION =================
if len(X_seq) == 0:
    st.warning("Not enough rows to make sequences (need > 30). Showing raw AQI only.")
    st.stop()

if model is None:
    # Demo prediction: smoothed + slight drift
    base = pd.Series(df_city["aqi"].values[LOOKBACK:])
    smooth = base.rolling(3, min_periods=1).mean()
    noise = np.random.normal(0, max(1.0, base.std() * 0.03), size=len(smooth))
    y_pred = (smooth + noise).values.reshape(-1, 1)
else:
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

df_pred = df_city.iloc[LOOKBACK:].copy()
df_pred["Predicted AQI"] = np.ravel(y_pred)

st.subheader(f"📈 Actual vs Predicted — {city} ({model_choice})")
fig = px.line(
    df_pred,
    x="date",
    y=["aqi", "Predicted AQI"],
    labels={"value": "AQI", "variable": "Series"},
)
st.plotly_chart(fig, use_container_width=True)

# ================= 7-DAY FORECAST =================
st.subheader("🔮 7-Day AQI Forecast")

if model is None:
    # Demo: random walk from last AQI
    last = float(df_city["aqi"].iloc[-1])
    steps = np.random.normal(0, max(2.0, df_city["aqi"].std() * 0.05), 7)
    future_aqi = np.clip(last + np.cumsum(steps), 0, None)
else:
    # Autoregressive window on scaled space
    window = X[-LOOKBACK:, :].copy()
    preds_scaled = []
    for _ in range(7):
        pred_scaled = model.predict(window[np.newaxis, :, :], verbose=0)[0][0]
        preds_scaled.append(pred_scaled)
        window = np.roll(window, -1, axis=0)
        # put the predicted AQI back into the last row’s target slot sensibly:
        # if 1D target, copy last feature vector and overwrite first feature
        if window.shape[1] > 0:
            window[-1, 0] = pred_scaled
    future_aqi = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()

future_dates = pd.date_range(df_city["date"].iloc[-1] + pd.Timedelta(days=1), periods=7)
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted AQI": future_aqi})

st.dataframe(forecast_df.style.format({"Predicted AQI": "{:.2f}"}))
fig2 = px.line(forecast_df, x="Date", y="Predicted AQI", labels={"Predicted AQI": "AQI"})
st.plotly_chart(fig2, use_container_width=True)

# ================= FOOTER =================
st.markdown("---")
note = "TensorFlow not detected — running demo mode." if model is None else "TensorFlow model active."
st.caption(f"Developed by **Deepesh Srivastava** · {note} 🌿")
