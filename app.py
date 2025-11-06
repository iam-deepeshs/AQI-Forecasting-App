import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Try to import TensorFlow (optional)
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    load_model = None
    TF_AVAILABLE = False

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="🌏 AQI Forecast Dashboard", layout="wide")

# ====================== CUSTOM STYLES ======================
st.markdown("""
    <style>
    /* Global background and font */
    body { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #f3f9f6, #eef6ff);
    }

    /* Header styling */
    .main-header {
        padding: 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #2e8b57, #00bfff);
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    /* Cards for metrics */
    .metric-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        text-align: center;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        justify-content: center;
        gap: 1rem;
    }

    /* Footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
<div class="main-header">
    <h1>🌏 India AQI Forecast Dashboard</h1>
    <p>Powered by LSTM & GRU Deep Learning Models · Interactive Visualization & Predictions</p>
</div>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Control Panel")
st.sidebar.write("Use this panel to select dataset, model, and city.")

uploaded_file = st.sidebar.file_uploader("📂 Upload AQI CSV file", type=["csv"])

# ====================== LOAD DATA ======================
@st.cache_data(show_spinner=False)
def load_csv_smart(uploaded):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        paths = ["data/city_day_small.csv", "AQI_India.csv"]
        path = next((p for p in paths if os.path.exists(p)), None)
        if not path:
            st.error("❌ No file uploaded and default dataset not found.")
            st.stop()
        df = pd.read_csv(path)
        st.sidebar.success(f"✅ Using default: {path}")

    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["city", "aqi"])
    return df

df = load_csv_smart(uploaded_file)

# ====================== CITY SELECTOR ======================
city_list = sorted(df["city"].unique())
city = st.sidebar.selectbox("🏙️ Select City", city_list)

# City Data
df_city = df[df["city"] == city].sort_values("date")

# ====================== MODEL LOADING ======================
model_choice = st.sidebar.radio("🤖 Choose Model", ["LSTM", "GRU"])

def load_keras_model_safe(model_choice):
    if not TF_AVAILABLE:
        return None, "TensorFlow not available, using demo forecast."
    model_path = os.path.join("models", f"{model_choice.lower()}_model.h5")
    if os.path.exists(model_path):
        try:
            return load_model(model_path), f"✅ {model_choice} model loaded!"
        except Exception as e:
            return None, f"⚠️ Error loading model: {e}"
    return None, "⚠️ Model file not found."

model, model_status = load_keras_model_safe(model_choice)
st.sidebar.info(model_status)

# ====================== FEATURE ENGINEERING ======================
features = [c for c in ["pm2.5", "pm10", "no2", "so2", "co", "o3", "nh3"] if c in df_city.columns]
if not features:
    df_city["pm2.5"] = df_city["aqi"]
    features = ["pm2.5"]

scaler_X, scaler_y = StandardScaler(), StandardScaler()
X = scaler_X.fit_transform(df_city[features])
y = scaler_y.fit_transform(df_city[["aqi"]])

def make_sequences(X, y, lookback=30):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

LOOKBACK = 30
X_seq, y_seq = make_sequences(X, y, LOOKBACK)

# ====================== PREDICTIONS ======================
if model is None:
    base = pd.Series(df_city["aqi"].values[LOOKBACK:])
    y_pred = base.rolling(3, min_periods=1).mean() + np.random.normal(0, 2, len(base))
else:
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

df_pred = df_city.iloc[LOOKBACK:].copy()
df_pred["Predicted AQI"] = y_pred

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["📈 AQI Trend", "🤖 Prediction", "🔮 7-Day Forecast"])

with tab1:
    st.subheader(f"AQI Trend — {city}")
    fig = px.line(df_city, x="date", y="aqi", title=f"AQI Levels in {city}", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Actual vs Predicted AQI ({model_choice})")
    fig2 = px.line(df_pred, x="date", y=["aqi", "Predicted AQI"], labels={"value": "AQI", "variable": "Type"})
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("7-Day AQI Forecast")
    if model is None:
        last = df_city["aqi"].iloc[-1]
        noise = np.random.normal(0, 1.5, 7)
        future_aqi = np.clip(last + np.cumsum(noise), 0, None)
    else:
        window = X[-LOOKBACK:, :]
        preds_scaled = []
        for _ in range(7):
            pred = model.predict(window[np.newaxis, :, :])[0][0]
            preds_scaled.append(pred)
            window = np.roll(window, -1, axis=0)
            window[-1, 0] = pred
        future_aqi = scaler_y.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(df_city["date"].iloc[-1] + pd.Timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted AQI": future_aqi})
    st.dataframe(forecast_df.style.format({"Predicted AQI": "{:.2f}"}))
    fig3 = px.area(forecast_df, x="Date", y="Predicted AQI", title="7-Day Forecast")
    st.plotly_chart(fig3, use_container_width=True)

# ====================== FOOTER ======================
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:gray;'>Developed by <b>Deepesh Srivastava</b> 🌿 | {model_status}</p>",
    unsafe_allow_html=True
)
