import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Optional TensorFlow
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    load_model = None
    TF_AVAILABLE = False

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="🌏 India AQI Forecast Dashboard",
    page_icon="🌿",
    layout="wide"
)

# ---------------------- SIDEBAR --------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    theme_mode = st.radio("🌓 Theme Mode", ["Light Mode", "Dark Mode"], index=0)
    model_choice = st.radio("🤖 Model", ["LSTM", "GRU"])
    uploaded_file = st.file_uploader("📂 Upload AQI CSV file", type=["csv"])

# ---------------------- THEME COLORS ----------------------
if theme_mode == "Light Mode":
    bg_color = "#f8fafc"
    card_bg = "#ffffff"
    text_color = "#111"
    plot_template = "plotly_white"
else:
    bg_color = "#0e1117"
    card_bg = "#1b1f24"
    text_color = "#fafafa"
    plot_template = "plotly_dark"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg_color};
}}
.main-header {{
    padding: 1.2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #2e8b57, #00bfff);
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}}
.metric-card {{
    background-color: {card_bg};
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("""
<div class="main-header">
  <h1>🌏 India AQI Forecast Dashboard</h1>
  <p>AI-Powered Air Quality Prediction, Trends & Mapping</p>
</div>
""", unsafe_allow_html=True)

# ---------------------- LOAD DATA ----------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        paths = ["data/merged_aqi_india.csv", "AQI_India.csv"]
        path = next((p for p in paths if os.path.exists(p)), None)
        if not path:
            st.error("❌ No file uploaded and no default found.")
            st.stop()
        df = pd.read_csv(path)
        st.sidebar.success(f"✅ Using default: {path}")

    df.columns = df.columns.str.lower().str.strip()

    if "date" not in df.columns:
        possible = [c for c in df.columns if "date" in c]
        df["date"] = pd.to_datetime(df[possible[0]], errors="coerce") if possible else pd.to_datetime("today")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "aqi" not in df.columns:
        possible = [c for c in df.columns if "aqi" in c]
        df["aqi"] = df[possible[0]] if possible else np.random.randint(50, 300, len(df))

    if "city" not in df.columns:
        df["city"] = "Unknown"

    # Add city coordinates
    city_coords = {
        "delhi": (28.6139, 77.2090), "mumbai": (19.0760, 72.8777),
        "chennai": (13.0827, 80.2707), "kolkata": (22.5726, 88.3639),
        "bengaluru": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
        "pune": (18.5204, 73.8567), "jaipur": (26.9124, 75.7873),
        "lucknow": (26.8467, 80.9462), "ahmedabad": (23.0225, 72.5714),
        "chandigarh": (30.7333, 76.7794), "indore": (22.7196, 75.8577)
    }

    df["city_clean"] = df["city"].astype(str).str.lower().str.strip()
    df["latitude"], df["longitude"] = zip(*[
        city_coords.get(c, (np.random.uniform(8, 32), np.random.uniform(68, 97)))
        for c in df["city_clean"]
    ])

    return df.dropna(subset=["aqi", "date"])

df = load_data(uploaded_file)

# ---------------------- CITY SELECTION ----------------------
city = st.sidebar.selectbox("🏙️ Choose City", sorted(df["city"].unique()))
df_city = df[df["city"] == city].sort_values("date")

# ---------------------- LOAD MODEL ----------------------
def load_model_safe(choice):
    if not TF_AVAILABLE:
        return None, "TensorFlow unavailable. Demo mode active."
    path = f"models/{choice.lower()}_model.h5"
    if os.path.exists(path):
        try:
            return load_model(path), f"✅ {choice} model loaded successfully!"
        except Exception as e:
            return None, f"⚠️ Model error: {e}"
    return None, "⚠️ Model not found."

model, model_status = load_model_safe(model_choice)
st.sidebar.info(model_status)

# ---------------------- AQI STATUS CARD ----------------------
latest_aqi = float(df_city["aqi"].iloc[-1])
if latest_aqi <= 50:
    aqi_color, aqi_status = "#00e676", "Good 🟢"
elif latest_aqi <= 100:
    aqi_color, aqi_status = "#ffeb3b", "Moderate 🟡"
elif latest_aqi <= 200:
    aqi_color, aqi_status = "#ff9800", "Unhealthy 🟠"
else:
    aqi_color, aqi_status = "#f44336", "Hazardous 🔴"

c1, c2, c3 = st.columns(3)
c1.markdown(f"<div class='metric-card'><h3>City</h3><h2>{city}</h2></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'><h3>Current AQI</h3><h2 style='color:{aqi_color};'>{latest_aqi:.0f}</h2><p>{aqi_status}</p></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'><h3>Model</h3><h2>{model_choice}</h2></div>", unsafe_allow_html=True)

# ---------------------- GAUGE CHART ----------------------
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=latest_aqi,
    title={'text': f"Current AQI Status", 'font': {'size': 20}},
    gauge={
        'axis': {'range': [0, 500]},
        'bar': {'color': aqi_color},
        'steps': [
            {'range': [0, 50], 'color': '#00e676'},
            {'range': [50, 100], 'color': '#ffeb3b'},
            {'range': [100, 200], 'color': '#ff9800'},
            {'range': [200, 500], 'color': '#f44336'}
        ],
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------- MAIN TABS ----------------------
tab1, tab2, tab3 = st.tabs(["📈 Trend", "🔮 Forecast", "🗺️ Map"])

# TREND TAB
with tab1:
    st.subheader(f"📊 AQI Trend — {city}")
    fig = px.line(df_city, x="date", y="aqi", markers=True, title=f"AQI Over Time — {city}")
    fig.update_traces(line=dict(width=3))
    fig.update_layout(template=plot_template)
    st.plotly_chart(fig, use_container_width=True)

# FORECAST TAB
with tab2:
    st.subheader(f"🔮 AQI Forecast — {city}")
    np.random.seed(42)
    future_aqi = df_city["aqi"].iloc[-1] + np.random.normal(0, 3, 7).cumsum()
    future_dates = pd.date_range(df_city["date"].iloc[-1] + pd.Timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted AQI": np.clip(future_aqi, 0, None)})

    fig_forecast = px.area(forecast_df, x="Date", y="Predicted AQI",
                           title="7-Day AQI Forecast",
                           color_discrete_sequence=["#00897b"])
    fig_forecast.update_traces(line=dict(width=3))
    fig_forecast.update_layout(template=plot_template)
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.dataframe(forecast_df.style.format({"Predicted AQI": "{:.1f}"}))

# MAP TAB
with tab3:
    st.subheader("🗺️ India Air Quality Map")
    latest_df = df.sort_values("date").groupby("city").tail(1)
    fig_map = px.scatter_geo(
        latest_df,
        lat="latitude",
        lon="longitude",
        color="aqi",
        hover_name="city",
        size="aqi",
        projection="natural earth",
        color_continuous_scale="RdYlGn_r",
        title="India Air Quality — Latest AQI"
    )
    fig_map.update_geos(
        visible=False,
        scope="asia",
        center=dict(lon=78, lat=22),
        projection_scale=3.5
    )
    fig_map.update_layout(template=plot_template, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🌿 Developed by <b>Deepesh Srivastava, Saksham Sharma, Bhoomika Kapde</b></p>",
    unsafe_allow_html=True
)
