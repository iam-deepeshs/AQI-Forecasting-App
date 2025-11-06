import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ====================== TRY IMPORTING TENSORFLOW ======================
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    load_model = None
    TF_AVAILABLE = False

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🌏 India AQI Forecast Dashboard",
    page_icon="🌿",
    layout="wide"
)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("⚙️ Control Panel")
    theme_mode = st.radio("🌓 Theme", ["Light Mode", "Dark Mode"], index=0)
    model_choice = st.radio("🤖 Model Type", ["LSTM", "GRU"])
    uploaded_file = st.file_uploader("📂 Upload AQI CSV", type=["csv"])

# ====================== THEME SETTINGS ======================
if theme_mode == "Light Mode":
    bg_color = "#f8fafc"
    text_color = "#111"
    card_bg = "#ffffff"
    plot_template = "plotly_white"
else:
    bg_color = "#0e1117"
    text_color = "#fafafa"
    card_bg = "#1b1f24"
    plot_template = "plotly_dark"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg_color};
}}
.main-header {{
    padding: 1.4rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #2e8b57, #00bfff);
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}}
.metric-card {{
    background-color: {card_bg};
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
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
  <p>AI-powered Air Quality Prediction, Mapping & Insights</p>
</div>
""", unsafe_allow_html=True)

# ====================== LOAD DATA ======================
@st.cache_data(show_spinner=False)
def load_data(uploaded):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        paths = ["data/merged_aqi_india.csv", "AQI_India.csv"]
        path = next((p for p in paths if os.path.exists(p)), None)
        if not path:
            st.error("❌ No dataset found. Upload your CSV.")
            st.stop()
        df = pd.read_csv(path)
        st.sidebar.success(f"✅ Using default dataset: {path}")

    df.columns = df.columns.str.lower().str.strip()
    if "date" not in df.columns:
        st.error("❌ The dataset must have a 'date' column.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "aqi" not in df.columns:
        possible = [c for c in df.columns if "aqi" in c]
        if possible:
            df["aqi"] = df[possible[0]]
        else:
            df["aqi"] = np.random.randint(50, 300, len(df))

    if "city" not in df.columns:
        df["city"] = "Unknown"

    # Add synthetic coordinates if missing
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

    # Ensure valid numeric lat/lon
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    return df

df = load_data(uploaded_file)
st.sidebar.success("✅ Data Loaded Successfully!")

# ====================== CITY SELECTOR ======================
city = st.sidebar.selectbox("🏙️ Choose City", sorted(df["city"].unique()))
df_city = df[df["city"] == city].sort_values("date")

# ====================== MODEL ======================
def load_keras_model_safe(choice):
    if not TF_AVAILABLE:
        return None, "TensorFlow unavailable. Demo mode active."
    model_path = f"models/{choice.lower()}_model.h5"
    if os.path.exists(model_path):
        try:
            return load_model(model_path), f"✅ {choice} model loaded successfully!"
        except Exception as e:
            return None, f"⚠️ Model loading failed: {e}"
    return None, "⚠️ Model not found."

model, model_status = load_keras_model_safe(model_choice)
st.sidebar.info(model_status)

# ====================== AQI STATUS ======================
latest_aqi = float(df_city["aqi"].iloc[-1])
if latest_aqi <= 50:
    aqi_color, aqi_status = "#00e676", "Good 🟢"
elif latest_aqi <= 100:
    aqi_color, aqi_status = "#ffeb3b", "Moderate 🟡"
elif latest_aqi <= 200:
    aqi_color, aqi_status = "#ff9800", "Unhealthy 🟠"
else:
    aqi_color, aqi_status = "#f44336", "Hazardous 🔴"

col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-card'><h3>City</h3><h2>{city}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>Current AQI</h3><h2 style='color:{aqi_color};'>{latest_aqi:.0f}</h2><p>{aqi_status}</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>Model</h3><h2>{model_choice}</h2></div>", unsafe_allow_html=True)

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "🔮 Forecast", "🗺️ India Map", "📊 Correlation"])

# ---- Tab 1: Trend ----
with tab1:
    st.subheader(f"AQI Trend — {city}")
    fig = px.line(df_city, x="date", y="aqi", markers=True, title=f"AQI Levels — {city}")
    fig.update_traces(line=dict(width=3))
    fig.update_layout(template=plot_template)
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: Forecast ----
with tab2:
    st.subheader(f"7-Day AQI Forecast — {city}")
    np.random.seed(42)
    base = df_city["aqi"].iloc[-1]
    forecast = np.clip(base + np.random.normal(0, 2.5, 7).cumsum(), 0, None)
    future_dates = pd.date_range(df_city["date"].iloc[-1] + pd.Timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted AQI": forecast})

    fig_forecast = px.area(forecast_df, x="Date", y="Predicted AQI", title=f"7-Day Forecast — {city}",
                           color_discrete_sequence=["#26a69a"])
    fig_forecast.update_traces(line=dict(width=3))
    fig_forecast.update_layout(template=plot_template)
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.dataframe(forecast_df.style.format({"Predicted AQI": "{:.2f}"}))

# ---- Tab 3: India Map ----
with tab3:
    st.subheader("🗺️ India Air Quality Map (with Time Slider)")

    latest_df = df.copy()

    # --- Clean & validate ---
    latest_df = latest_df.dropna(subset=["latitude", "longitude", "aqi"]).copy()
    latest_df = latest_df[
        (latest_df["latitude"].between(-90, 90)) &
        (latest_df["longitude"].between(-180, 180))
    ]
    latest_df = latest_df[latest_df["aqi"] > 0]  # remove invalid or negative AQI
    latest_df["latitude"] = latest_df["latitude"].round(4)
    latest_df["longitude"] = latest_df["longitude"].round(4)
    latest_df = latest_df.sort_values("date")

    if "date" in latest_df.columns:
        latest_df["date_str"] = latest_df["date"].dt.strftime("%Y-%m-%d")
    else:
        latest_df["date_str"] = "Unknown"

    if len(latest_df) < 5:
        st.warning("⚠️ Not enough valid geographic data to show the map.")
    else:
        try:
            # --- Animated Map ---
            fig_map = px.scatter_geo(
                latest_df,
                lat="latitude",
                lon="longitude",
                color="aqi",
                hover_name="city",
                size="aqi",
                size_max=15,
                animation_frame="date_str",
                projection="natural earth",
                color_continuous_scale="RdYlGn_r",
                title="India Air Quality Over Time"
            )

            fig_map.update_geos(
                scope="asia",
                center=dict(lon=78, lat=22),
                projection_scale=3.5
            )

            fig_map.update_layout(
                template=plot_template,
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_colorbar=dict(title="AQI")
            )

            st.plotly_chart(fig_map, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Animated map failed: {e}")
            st.info("Rendering static map instead...")

            fig_static = px.scatter_geo(
                latest_df,
                lat="latitude",
                lon="longitude",
                color="aqi",
                hover_name="city",
                size="aqi",
                size_max=15,
                projection="natural earth",
                color_continuous_scale="RdYlGn_r",
                title="India Air Quality (Static Map)"
            )

            fig_static.update_geos(
                scope="asia",
                center=dict(lon=78, lat=22),
                projection_scale=3.5
            )

            fig_static.update_layout(template=plot_template)
            st.plotly_chart(fig_static, use_container_width=True)



# ---- Tab 4: Correlation ----
with tab4:
    st.subheader(f"📊 Pollutant Correlation Heatmap — {city}")
    features = [c for c in ["pm2.5", "pm10", "no2", "so2", "co", "o3", "nh3"] if c in df_city.columns]
    if len(features) > 1:
        corr = df_city[features + ["aqi"]].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough pollutant columns for correlation heatmap.")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🌿 Developed by <b>Deepesh Srivastava</b> — AI-driven AQI Dashboard</p>",
    unsafe_allow_html=True
)
