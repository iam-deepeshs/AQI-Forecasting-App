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
st.set_page_config(page_title="🌏 India AQI Forecast Dashboard", layout="wide", page_icon="🌿")

# ====================== SIDEBAR SETTINGS ======================
st.sidebar.title("⚙️ Control Panel")
theme_mode = st.sidebar.radio("🌓 Theme", ["Light Mode", "Dark Mode"], index=0)

# ====================== CSS THEME ======================
if theme_mode == "Light Mode":
    bg_color = "#f9fafb"
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
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #2e8b57, #00bfff);
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    .metric-card {{
        background-color: {card_bg};
        padding: 1rem;
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
    <h1>🌏 India Air Quality Forecast Dashboard</h1>
    <p>AI-powered AQI predictions using LSTM/GRU models — with live visualizations, forecasts & geospatial insights</p>
</div>
""", unsafe_allow_html=True)

# ====================== DATA LOADING ======================
DATA_PATH = "Final_Merged_AQI_with_LatLon.csv"

@st.cache_data(show_spinner=False)
def load_data(path):
    if not os.path.exists(path):
        st.error("❌ Dataset not found. Please upload your data file.")
        st.stop()
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["city", "aqi"])
    return df

df = load_data(DATA_PATH)
st.sidebar.success("✅ Data Loaded Successfully!")

# ====================== CITY SELECTOR ======================
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

# ====================== DATA PREP ======================
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

# ====================== PREDICTIONS ======================
if model is None:
    y_pred = pd.Series(df_city["aqi"].values[LOOKBACK:]).rolling(3, min_periods=1).mean() + np.random.normal(0, 2, len(df_city) - LOOKBACK)
else:
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

df_pred = df_city.iloc[LOOKBACK:].copy()
df_pred["Predicted AQI"] = y_pred

# ====================== AQI STATUS ======================
latest_aqi = float(df_city["aqi"].iloc[-1])
if latest_aqi <= 50:
    aqi_color, status = "#00e676", "Good 🟢"
elif latest_aqi <= 100:
    aqi_color, status = "#ffeb3b", "Moderate 🟡"
elif latest_aqi <= 200:
    aqi_color, status = "#ff9800", "Unhealthy 🟠"
else:
    aqi_color, status = "#f44336", "Very Unhealthy 🔴"

col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-card'><h3 style='color:{aqi_color};'>Current AQI</h3><h2 style='color:{aqi_color};'>{latest_aqi:.0f}</h2><p>{status}</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>City</h3><h2>{city}</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>Model</h3><h2>{model_choice}</h2></div>", unsafe_allow_html=True)

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trend", "🤖 Prediction", "🔮 Forecast", "🗺️ India Map", "📅 Monthly Trend", "📊 Correlation Heatmap"
])

# ====================== TAB 1 — Trend ======================
with tab1:
    st.subheader(f"AQI Trend for {city}")
    fig = px.line(df_city, x="date", y="aqi", title=f"AQI Levels — {city}", markers=True)
    fig.update_traces(line=dict(width=3))
    fig.update_layout(template="plotly_white" if theme_mode == "Light Mode" else "plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ====================== TAB 2 — Prediction ======================
with tab2:
    st.subheader(f"Actual vs Predicted AQI ({model_choice})")
    fig2 = px.line(df_pred, x="date", y=["aqi", "Predicted AQI"], labels={"value": "AQI", "variable": "Type"})
    fig2.update_layout(template="plotly_white" if theme_mode == "Light Mode" else "plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

# ====================== TAB 3 — Forecast ======================
with tab3:
    st.subheader("🔮 7-Day Forecast")
    if model is None:
        noise = np.random.normal(0, 1.5, 7)
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
    fig3 = px.area(forecast_df, x="Date", y="Predicted AQI", title=f"7-Day Forecast — {city}")
    st.plotly_chart(fig3, use_container_width=True)

# ====================== TAB 4 — India Map ======================
with tab4:
    st.subheader("🗺️ India Air Quality Map")
    df_map = df.dropna(subset=["latitude", "longitude", "aqi"])
    df_map = df_map[df_map["latitude"].between(6, 38) & df_map["longitude"].between(68, 98)]
    if not df_map.empty:
        latest_df = df_map.sort_values("date").groupby("city").tail(1)
        latest_df = latest_df[latest_df["aqi"].notna() & latest_df["latitude"].notna()]
        fig_map = px.scatter_geo(
            latest_df,
            lat="latitude",
            lon="longitude",
            color="aqi",
            hover_name="city",
            size=latest_df["aqi"].fillna(10),
            projection="natural earth",
            color_continuous_scale="RdYlGn_r",
            title="India Air Quality — Latest AQI",
        )
        fig_map.update_layout(
            geo=dict(scope="asia", center=dict(lon=78, lat=22), projection_scale=3.2),
            template="plotly_white" if theme_mode == "Light Mode" else "plotly_dark"
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("⚠️ No valid latitude/longitude data available after cleaning.")

# ====================== TAB 5 — Monthly Trend ======================
with tab5:
    st.subheader(f"📅 Monthly AQI Trends — {city}")
    df_city["month"] = df_city["date"].dt.month_name()
    monthly = df_city.groupby("month")["aqi"].mean().reindex([
        'January','February','March','April','May','June','July','August','September','October','November','December'
    ])
    fig_season = px.bar(monthly, x=monthly.index, y=monthly.values,
                        color=monthly.values, color_continuous_scale="RdYlGn_r",
                        title=f"Average Monthly AQI — {city}")
    fig_season.update_layout(template="plotly_white" if theme_mode == "Light Mode" else "plotly_dark")
    st.plotly_chart(fig_season, use_container_width=True)

# ====================== TAB 6 — Correlation Heatmap ======================
with tab6:
    st.subheader("📊 Pollutant Correlation Heatmap")
    if len(features) > 1:
        corr = df_city[features + ["aqi"]].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough pollutant columns to generate a correlation heatmap.")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:gray;'>🌿 Developed by <b>Deepesh Srivastava</b> · {model_status}</p>",
    unsafe_allow_html=True
)
