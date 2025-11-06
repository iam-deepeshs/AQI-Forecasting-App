import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Try TensorFlow if available
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    load_model = None
    TF_AVAILABLE = False

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="🌏 AQI Forecast Dashboard", layout="wide", page_icon="🌿")

# ====================== SIDEBAR SETTINGS ======================
st.sidebar.title("⚙️ Control Panel")
theme_mode = st.sidebar.radio("🌓 Theme", ["Light Mode", "Dark Mode"], index=0)

# ====================== DYNAMIC CSS THEMES ======================
if theme_mode == "Light Mode":
    bg_color = "#f7fafc"
    text_color = "#111"
    card_bg = "#ffffff"
else:
    bg_color = "#0e1117"
    text_color = "#fafafa"
    card_bg = "#1b1f24"

st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
    }}
    .main-header {{
        padding: 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #2e8b57, #00bfff);
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    .metric-card {{
        background-color: {card_bg};
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        text-align: center;
        margin: 10px;
    }}
    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
<div class="main-header">
    <h1>🌏 India AQI Forecast Dashboard</h1>
    <p>AI-powered AQI predictions using LSTM/GRU models — Visual insights, forecasts & mapping</p>
</div>
""", unsafe_allow_html=True)

# ====================== DATA UPLOAD ======================
uploaded_file = st.sidebar.file_uploader("📂 Upload AQI CSV file", type=["csv"])

@st.cache_data(show_spinner=False)
def load_data(uploaded):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        paths = ["data/merged_aqi_india.csv", "data/city_day_small.csv", "AQI_India.csv"]
        path = next((p for p in paths if os.path.exists(p)), None)
        if not path:
            st.error("❌ No file uploaded and default dataset not found.")
            st.stop()
        df = pd.read_csv(path)
        st.sidebar.success(f"✅ Using default: {path}")

    df.columns = df.columns.str.lower().str.strip()

    # ------------------------- Fix missing columns -------------------------
    if "date" not in df.columns:
        possible_dates = [c for c in df.columns if "date" in c]
        if possible_dates:
            df["date"] = pd.to_datetime(df[possible_dates[0]], errors="coerce")
        else:
            st.error("❌ No date column found in dataset.")
            st.stop()
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "city" not in df.columns:
        st.error("❌ 'city' column missing. Please include a 'city' column.")
        st.stop()

    if "aqi" not in df.columns:
        possible_aqi = [c for c in df.columns if "aqi" in c]
        if possible_aqi:
            df["aqi"] = df[possible_aqi[0]]
        else:
            st.error("❌ No AQI column found in dataset.")
            st.stop()

    df = df.dropna(subset=["city", "aqi"])

    # ------------------------- Add Latitude/Longitude -------------------------
    city_coords = {
        "delhi": (28.6139, 77.2090), "new delhi": (28.6139, 77.2090),
        "mumbai": (19.0760, 72.8777), "chennai": (13.0827, 80.2707),
        "kolkata": (22.5726, 88.3639), "bengaluru": (12.9716, 77.5946),
        "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
        "ahmedabad": (23.0225, 72.5714), "pune": (18.5204, 73.8567),
        "jaipur": (26.9124, 75.7873), "lucknow": (26.8467, 80.9462),
        "kanpur": (26.4499, 80.3319), "patna": (25.5941, 85.1376),
        "indore": (22.7196, 75.8577), "bhopal": (23.2599, 77.4126),
        "nagpur": (21.1458, 79.0882), "chandigarh": (30.7333, 76.7794),
        "varanasi": (25.3176, 82.9739), "surat": (21.1702, 72.8311),
        "visakhapatnam": (17.6868, 83.2185), "coimbatore": (11.0168, 76.9558),
        "noida": (28.5355, 77.3910), "gurugram": (28.4595, 77.0266),
        "nashik": (19.9975, 73.7898), "vadodara": (22.3072, 73.1812),
        "mysuru": (12.2958, 76.6394), "ranchi": (23.3441, 85.3096),
        "raipur": (21.2514, 81.6296), "guwahati": (26.1445, 91.7362),
        "thiruvananthapuram": (8.5241, 76.9366), "madurai": (9.9252, 78.1198),
        "agra": (27.1767, 78.0081), "meerut": (28.9845, 77.7064),
        "amritsar": (31.6340, 74.8723), "dehradun": (30.3165, 78.0322),
        "shimla": (31.1048, 77.1734), "srinagar": (34.0837, 74.7973)
    }

    df["city_clean"] = df["city"].astype(str).str.lower().str.strip()

    def get_coords(name):
        for known in city_coords.keys():
            if known in name:  # partial match
                return city_coords[known]
        # fallback: random coordinates within India
        return (np.random.uniform(8, 32), np.random.uniform(68, 97))

    df["latitude"], df["longitude"] = zip(*df["city_clean"].map(get_coords))

    if df["latitude"].isna().all():
        st.warning("⚠️ No valid latitude/longitude data available even after processing.")
    else:
        st.sidebar.success("✅ Data Loaded Successfully!")

    return df

# Load dataset
df = load_data(uploaded_file)

# ====================== CITY SELECTION ======================
city_list = sorted(df["city"].unique())
city = st.sidebar.selectbox("🏙️ Choose City", city_list)
df_city = df[df["city"] == city].sort_values("date")

# ====================== MODEL CHOICE ======================
model_choice = st.sidebar.radio("🤖 Model Type", ["LSTM", "GRU"])

def load_keras_model_safe(choice):
    if not TF_AVAILABLE:
        return None, "TensorFlow unavailable. Demo mode active."
    model_path = os.path.join("models", f"{choice.lower()}_model.h5")
    if os.path.exists(model_path):
        try:
            return load_model(model_path), f"✅ {choice} model loaded successfully!"
        except Exception as e:
            return None, f"⚠️ Error loading model: {e}"
    return None, "⚠️ Model not found."

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
X_seq, y_seq = make_sequences(X, y)

# ====================== AQI TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "🔮 Forecast", "🗺️ India Map", "📊 Correlation"])

# Trend Chart
with tab1:
    st.subheader(f"AQI Trend for {city}")
    fig = px.line(df_city, x="date", y="aqi", title=f"{city} — AQI Trend", markers=True)
    fig.update_layout(template="plotly_white" if theme_mode == "Light Mode" else "plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Forecast Tab
with tab2:
    st.subheader("7-Day AQI Forecast")
    if model is None:
        noise = np.random.normal(0, 2, 7)
        base = df_city["aqi"].iloc[-1]
        future_aqi = np.clip(base + np.cumsum(noise), 0, None)
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
    fig_forecast = px.area(forecast_df, x="Date", y="Predicted AQI", title=f"7-Day Forecast — {city}")
    fig_forecast.update_layout(template="plotly_white" if theme_mode == "Light Mode" else "plotly_dark")
    st.plotly_chart(fig_forecast, use_container_width=True)

# India Map
with tab4:
    st.subheader("🌏 India Air Quality Map")
    latest_df = df.sort_values("date").groupby("city").tail(1)
    if "latitude" in latest_df.columns and "longitude" in latest_df.columns:
        fig_map = px.scatter_geo(
            latest_df,
            lat="latitude",
            lon="longitude",
            color="aqi",
            hover_name="city",
            size="aqi",
            projection="natural earth",
            color_continuous_scale="RdYlGn_r",
            title="India Air Quality — Latest AQI",
        )
        fig_map.update_layout(
            geo=dict(scope="asia", center=dict(lon=78, lat=22), projection_scale=3.5),
            template="plotly_white" if theme_mode == "Light Mode" else "plotly_dark"
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("⚠️ No valid latitude/longitude data found.")

# Correlation Heatmap
with tab3:
    st.subheader("📊 Pollutant Correlation Heatmap")
    if len(features) > 1:
        corr = df_city[features + ["aqi"]].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough pollutant columns to generate a correlation heatmap.")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🌿 Developed by <b>Deepesh Srivastava, Saksham Sharma, Bhoomika Kapde</b></p>",
    unsafe_allow_html=True
)
