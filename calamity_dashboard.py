
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Natural Calamity Early Warning System",
    page_icon="Alert",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4e 100%); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 3000
    data = []
    for _ in range(n):
        seismic_activity  = np.random.uniform(0, 9.5)
        sea_level_change  = np.random.uniform(-2, 5)
        rainfall_mm       = np.random.uniform(0, 300)
        wind_speed_kmh    = np.random.uniform(0, 300)
        soil_moisture     = np.random.uniform(0, 100)
        slope_angle_deg   = np.random.uniform(0, 60)
        pressure_hpa      = np.random.uniform(870, 1020)
        ocean_temp_c      = np.random.uniform(10, 50)
        river_water_level = np.random.uniform(0, 15)
        humidity_pct      = np.random.uniform(5, 100)
        temperature_c     = np.random.uniform(10, 50)
        lightning_index   = np.random.uniform(0, 10)
        snow_depth_cm     = np.random.uniform(0, 100)
        visibility_km     = np.random.uniform(0, 50)

        if seismic_activity > 6.5 and sea_level_change > 2:
            label = "Tsunami"
        elif rainfall_mm > 150 and river_water_level > 10:
            label = "Heavy Rainfall"
        elif wind_speed_kmh > 120 and pressure_hpa < 950:
            label = "Cyclone"
        elif soil_moisture > 75 and slope_angle_deg > 35 and rainfall_mm > 80:
            label = "Landslide"
        elif temperature_c > 42 and humidity_pct < 20:
            label = "Heatwave"
        elif rainfall_mm < 5 and soil_moisture < 15 and humidity_pct < 20:
            label = "Drought"
        elif lightning_index > 7 and rainfall_mm > 50 and wind_speed_kmh > 60:
            label = "Thunderstorm"
        elif wind_speed_kmh > 180 and pressure_hpa < 960 and rainfall_mm < 30:
            label = "Tornado"
        elif snow_depth_cm > 50 and temperature_c < 0 and wind_speed_kmh > 60:
            label = "Blizzard"
        elif seismic_activity > 5.0 and sea_level_change < 0.5:
            label = "Earthquake"
        else:
            label = "No Calamity"

        data.append([seismic_activity, sea_level_change, rainfall_mm,
                     wind_speed_kmh, soil_moisture, slope_angle_deg,
                     pressure_hpa, ocean_temp_c, river_water_level,
                     humidity_pct, temperature_c, lightning_index,
                     snow_depth_cm, visibility_km, label])

    cols = ["seismic_activity","sea_level_change","rainfall_mm",
            "wind_speed_kmh","soil_moisture","slope_angle_deg",
            "pressure_hpa","ocean_temp_c","river_water_level",
            "humidity_pct","temperature_c","lightning_index",
            "snow_depth_cm","visibility_km","calamity_type"]

    df  = pd.DataFrame(data, columns=cols)
    le  = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["calamity_type"])
    X   = df.drop(columns=["calamity_type","label_encoded"])
    y   = df["label_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, le

def fetch_weather(city, api_key):
    try:
        url      = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        d        = response.json()
        if response.status_code != 200:
            return None
        rainfall_mm       = d.get("rain", {}).get("1h", 0.0)
        wind_speed_kmh    = round(d["wind"]["speed"] * 3.6, 2)
        pressure_hpa      = d["main"]["pressure"]
        humidity_pct      = d["main"]["humidity"]
        ocean_temp_c      = d["main"]["temp"]
        temperature_c     = d["main"]["temp"]
        visibility_km     = d.get("visibility", 10000) / 1000
        seismic_activity  = np.random.uniform(0, 2.5)
        sea_level_change  = np.random.uniform(-0.3, 0.8)
        soil_moisture     = min(100, humidity_pct * 0.85 + rainfall_mm * 0.5)
        slope_angle_deg   = np.random.uniform(10, 40)
        river_water_level = min(15, rainfall_mm * 0.1 + np.random.uniform(1, 4))
        lightning_index   = np.random.uniform(0, 10) if rainfall_mm > 20 else np.random.uniform(0, 3)
        snow_depth_cm     = np.random.uniform(0, 10) if temperature_c < 2 else 0.0
        return {
            "seismic_activity":  seismic_activity,
            "sea_level_change":  sea_level_change,
            "rainfall_mm":       rainfall_mm,
            "wind_speed_kmh":    wind_speed_kmh,
            "soil_moisture":     soil_moisture,
            "slope_angle_deg":   slope_angle_deg,
            "pressure_hpa":      pressure_hpa,
            "ocean_temp_c":      ocean_temp_c,
            "river_water_level": river_water_level,
            "humidity_pct":      humidity_pct,
            "temperature_c":     temperature_c,
            "lightning_index":   lightning_index,
            "snow_depth_cm":     snow_depth_cm,
            "visibility_km":     visibility_km
        }
    except:
        return None

ALERTS = {
    "Tsunami":        ("RED ALERT",    "#d32f2f", "EVACUATE COASTAL AREAS!"),
    "Heavy Rainfall": ("ORANGE ALERT", "#e65100", "MOVE TO HIGHER GROUND!"),
    "Cyclone":        ("RED ALERT",    "#d32f2f", "SEEK SHELTER IMMEDIATELY!"),
    "Landslide":      ("ORANGE ALERT", "#e65100", "VACATE HILLY AREAS!"),
    "Heatwave":       ("ORANGE ALERT", "#e65100", "STAY INDOORS! DRINK WATER!"),
    "Drought":        ("YELLOW ALERT", "#f9a825", "SAVE WATER!"),
    "Thunderstorm":   ("ORANGE ALERT", "#e65100", "AVOID OPEN AREAS!"),
    "Tornado":        ("RED ALERT",    "#d32f2f", "TAKE COVER UNDERGROUND!"),
    "Blizzard":       ("RED ALERT",    "#d32f2f", "STAY INDOORS!"),
    "Earthquake":     ("RED ALERT",    "#d32f2f", "DROP COVER HOLD ON!"),
    "No Calamity":    ("ALL CLEAR",    "#2e7d32", "Conditions Normal.")
}

CITY_COORDS = {
    "Chennai":     (13.0827, 80.2707),
    "Mumbai":      (19.0760, 72.8777),
    "Delhi":       (28.6139, 77.2090),
    "Kolkata":     (22.5726, 88.3639),
    "Bangalore":   (12.9716, 77.5946),
    "Hyderabad":   (17.3850, 78.4867),
    "Kochi":       (9.9312,  76.2673),
    "Vizag":       (17.6868, 83.2185),
    "Bhubaneswar": (20.2961, 85.8245),
    "Port Blair":  (11.6234, 92.7265)
}

st.markdown("<h1 style='text-align:center;color:white;'>Natural Calamity Early Warning System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#90CAF9;'>Live Detection - 10 Cities - 10 Calamity Types</p>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #1565C0;'>", unsafe_allow_html=True)

st.sidebar.title("Settings")
api_key       = st.sidebar.text_input("OpenWeatherMap API Key", type="password")
selected_city = st.sidebar.selectbox("Select City", list(CITY_COORDS.keys()))
auto_refresh  = st.sidebar.checkbox("Auto Refresh Every 60 Seconds", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**All Cities:**")
for city in CITY_COORDS.keys():
    st.sidebar.markdown(f"- {city}")

if not api_key:
    st.warning("Please enter your OpenWeatherMap API Key in the sidebar to start!")
    st.stop()

model, le = train_model()

if st.button("Scan All Cities Now", use_container_width=True):
    st.markdown("### Live Detection Results")
    results  = []
    col_idx  = 0
    progress = st.progress(0)
    status   = st.empty()
    city_cols = st.columns(2)

    for idx, (city, coords) in enumerate(CITY_COORDS.items()):
        status.text(f"Scanning {city}...")
        sensor = fetch_weather(city, api_key)

        if sensor:
            input_df   = pd.DataFrame([sensor])
            pred_code  = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            calamity   = le.inverse_transform([pred_code])[0]
            confidence = pred_proba.max() * 100
            level, color, action = ALERTS[calamity]

            results.append({
                "city":       city,
                "calamity":   calamity,
                "confidence": confidence,
                "level":      level,
                "temp":       sensor["temperature_c"],
                "wind":       sensor["wind_speed_kmh"],
                "rain":       sensor["rainfall_mm"],
                "humidity":   sensor["humidity_pct"],
                "pressure":   sensor["pressure_hpa"]
            })

            with city_cols[col_idx % 2]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05);border:2px solid {color};
                            border-radius:10px;padding:15px;margin:8px 0;">
                    <h3 style="color:white;margin:0;">{city}</h3>
                    <h4 style="color:{color};margin:5px 0;">{calamity} - {level}</h4>
                    <p style="color:#90CAF9;margin:0;">Confidence: {confidence:.1f}%</p>
                    <p style="color:white;margin:0;">Action: {action}</p>
                    <hr style="border:0.5px solid rgba(255,255,255,0.2);">
                    <p style="color:#B0BEC5;margin:0;">
                        Temp: {sensor["temperature_c"]:.1f}C |
                        Wind: {sensor["wind_speed_kmh"]:.1f} km/h |
                        Rain: {sensor["rainfall_mm"]:.1f} mm
                    </p>
                </div>
                """, unsafe_allow_html=True)
            col_idx += 1

        progress.progress((idx + 1) / len(CITY_COORDS))
        time.sleep(1)

    status.text("Scan complete!")
    progress.empty()

    if results:
        st.markdown("---")
        st.markdown("### Dashboard Charts")
        results_df = pd.DataFrame(results)
        chart_cols = st.columns(2)

        with chart_cols[0]:
            fig1 = px.bar(
                results_df, x="city", y="confidence",
                color="level",
                color_discrete_map={
                    "RED ALERT":    "#d32f2f",
                    "ORANGE ALERT": "#e65100",
                    "YELLOW ALERT": "#f9a825",
                    "ALL CLEAR":    "#2e7d32"
                },
                title="Confidence % by City",
                template="plotly_dark"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with chart_cols[1]:
            fig2 = px.bar(
                results_df, x="city", y="temp",
                color="temp",
                color_continuous_scale="RdYlGn_r",
                title="Temperature by City (C)",
                template="plotly_dark"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with chart_cols[0]:
            fig3 = px.bar(
                results_df, x="city", y="wind",
                color="wind",
                color_continuous_scale="Blues",
                title="Wind Speed by City (km/h)",
                template="plotly_dark"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with chart_cols[1]:
            fig4 = px.bar(
                results_df, x="city", y="humidity",
                color="humidity",
                color_continuous_scale="Teal",
                title="Humidity by City (%)",
                template="plotly_dark"
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.markdown("### Detection Summary Table")
        display_df = results_df[["city","calamity","level","confidence","temp","wind","rain","humidity"]].copy()
        display_df.columns = ["City","Calamity","Alert Level","Confidence %","Temp C","Wind km/h","Rain mm","Humidity %"]
        st.dataframe(display_df, use_container_width=True)

if auto_refresh:
    time.sleep(60)
    st.rerun()

st.markdown("<hr><p style='text-align:center;color:#546E7A;'>Natural Calamity Early Warning System</p>", unsafe_allow_html=True)
