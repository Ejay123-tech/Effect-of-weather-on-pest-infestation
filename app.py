import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import altair as alt
import folium
from folium.plugins import TimestampedGeoJson
import streamlit.components.v1 as components

# -------------------- CONFIG --------------------
st.set_page_config(page_title="🌽 Maize Pest Insights & Predictor", layout="wide")

# Predefined season mapping and list
mid_month = {
    'Jan–March': 2,
    'Feb–April': 3,
    'March–May': 4,
    'Aug–Oct': 9,
    'Sept–Dec': 10,
    'Oct–Nov': 10
}
SEASONS = list(mid_month.keys())

# Trigger options
def default_triggers():
    return [
        "Dry spells followed by rainfall",
        "Warm dry spells",
        "Heavy rains",
        "Cross-border swarm movement",
        "Dense weedy fields",
        "Night temps > 18°C"
    ]

# Load pretrained model and artifacts
try:
    model = joblib.load("pest_severity_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("Required model/vectorizer files not found. Ensure 'pest_severity_model.pkl', 'feature_names.pkl', and 'tfidf_vectorizer.pkl' are present.")
    st.stop()

severity_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
API_KEY = "091fd5c1ab03ae28846c1748ea358f97"  # your OpenWeatherMap API key

# -------------------- UTILITIES --------------------
@st.cache_data
def load_pest_data(path='maize_pest_dataset.csv'):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df['season_key'] = df['season/month']
    df['month_num'] = df['season_key'].map(mid_month).fillna(6).astype(int)
    df['date'] = pd.to_datetime(dict(year=2025, month=df['month_num'], day=15))
    sev_map = {"low":1, "medium":2, "high":3, "very high":4}
    df['severity_num'] = df['severity_level'].str.lower().map(sev_map)
    df.dropna(subset=['date','severity_num','location/region'], inplace=True)
    return df

@st.cache_data
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            return {
                "name": data['name'],
                "lat": data['coord']['lat'],
                "lon": data['coord']['lon'],
                "temp_max": data['main']['temp_max'],
                "temp_min": data['main']['temp_min'],
                "rainfall": data.get('rain', {}).get('1h', 0.0),
                "humidity": data['main']['humidity'],
                "wind_speed": data['wind']['speed'],
                "weather": data['weather'][0]['description'].capitalize()
            }
    except Exception:
        return None
    return None

def get_region(lat, lon):
    if -4 <= lat <= 4 and 33 <= lon <= 42:
        if lat < 0:
            if lon > 38: return "Coastal"
            if lon < 36: return "Western"
            return "Nyanza"
        if lon < 37: return "Rift Valley"
        if lat < 1: return "Central"
        return "Eastern"
    return "ASAL areas"

# Load data & prepare visuals
df = load_pest_data()
weekly = (
    df.set_index('date')
      .groupby('location/region')['severity_num']
      .resample('W')
      .mean()
      .reset_index(name='avg_severity')
)
np.random.seed(0)
weekly['humidity'] = 50 + 30 * np.sin(2 * np.pi * (weekly['date'].dt.month / 12)) + np.random.randn(len(weekly)) * 5

# Chart builders
def build_dual_chart(data):
    sev_line = alt.Chart(data).mark_line().encode(x='date:T', y='avg_severity:Q', color='location/region:N')
    hum_line = alt.Chart(data).mark_line(strokeDash=[5,5]).encode(x='date:T', y='humidity:Q', color='location/region:N')
    return alt.layer(sev_line, hum_line).resolve_scale(y='independent').properties(width=700, height=300)

def build_heatmap(data):
    heat = data.copy()
    heat['month'] = heat['date'].dt.strftime('%b')
    agg = heat.groupby(['location/region','month'])['severity_num'].mean().reset_index()
    return alt.Chart(agg).mark_rect().encode(
        x=alt.X('month:O', sort=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
        y='location/region:N', color='severity_num:Q'
    ).properties(width=600, height=200)

# Folium map builder
def build_hotspot_map(data):
    coords = {'Central':(-0.9,37.1),'Eastern':(-0.3,37.7),'Western Kenya':(0,35)}
    m = folium.Map(location=[-1.3,36.8], zoom_start=6)
    features = []
    for _, r in data.iterrows():
        loc, radius = r['location/region'], r['avg_severity']*4
        if loc in coords:
            lat, lon = coords[loc]
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': {'time': r['date'].strftime('%Y-%m-%d'), 'style': {'radius': radius, 'color': 'red'}}
            })
    TimestampedGeoJson({'type': 'FeatureCollection', 'features': features}, period='P1W', auto_play=False).add_to(m)
    return m
hotspot_map = build_hotspot_map(weekly)

# -------------------- STREAMLIT UI --------------------
st.title("🌽 Maize Pest Insights & Predictor")
tabs = st.tabs(["Predictor", "Trends & Humidity", "Heatmap", "Hotspots Map"])

# Predictor Tab
with tabs[0]:
    st.header("AI Maize Pest Severity Predictor")
    with st.form("predict_form"):
        city = st.text_input("📍 Enter your city/county", "Nyeri")
        season = st.selectbox("📅 Season", SEASONS, index=SEASONS.index('March–May'))
        stage = st.selectbox("🌱 Crop Stage", ["Seedling","Vegetative","Tasseling","Silking","Grain filling","Maturity"])
        pest = st.selectbox("🦟 Pest", ["Fall Armyworm","Corn Earworm","Locust"])
        trigger = st.selectbox("⚡ Trigger Condition", default_triggers())
        custom = st.checkbox("Custom trigger?")
        if custom:
            trigger = st.text_area("Enter custom trigger", trigger)
        submit = st.form_submit_button("🔍 Predict")

    if submit:
        weather = get_weather(city)
        if weather:
            region = get_region(weather['lat'], weather['lon'])
            base = {col: 0 for col in feature_names}
            base.update({f"Pest_{pest}": 1, f"Season/Month_{season}": 1, f"Crop Stage Affected_{stage}": 1, f"Location/Region_{region}": 1})
            vec = tfidf.transform([trigger]).toarray()[0]
            for i, w in enumerate(tfidf.get_feature_names_out()):
                col = f"TFIDF_{w}"
                if col in base:
                    base[col] = vec[i]
            input_df = pd.DataFrame([base])[feature_names]
            pred = model.predict(input_df)[0]
            sev = severity_map[pred]
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Location & Weather")
                st.write(f"**{weather['name']}** ({region})")
                st.write(f"Temp: {weather['temp_min']}–{weather['temp_max']}°C | Humidity: {weather['humidity']}% | Rainfall: {weather['rainfall']} mm")
                st.map(pd.DataFrame([{'lat': weather['lat'], 'lon': weather['lon']}]), zoom=8)
            with c2:
                st.subheader("Prediction")
                st.write(f"Pest: **{pest}**")
                st.write(f"Trigger: *{trigger}*")
                st.write(f"Severity: **{sev}**")
                if sev == "Very High": st.error("⚠️ Immediate action required!")
                elif sev == "High": st.warning("🔶 High risk. Apply measures.")
                elif sev == "Medium": st.info("🟡 Medium risk.")
                else: st.success("🟢 Low risk.")
        else:
            st.error("Could not fetch weather. Check city name or API key.")

# Trends & Humidity Tab
with tabs[1]:
    st.header("📈 Severity vs. Humidity Over Time")
    st.altair_chart(build_dual_chart(weekly), use_container_width=True)

# Heatmap Tab
with tabs[2]:
    st.header("🌡️ Avg Severity by Region & Month")
    st.altair_chart(build_heatmap(df), use_container_width=True)

# Hotspots Map Tab
with tabs[3]:
    st.header("🌍 Animated Hotspots Map")
    components.html(hotspot_map._repr_html_(), height=500)
