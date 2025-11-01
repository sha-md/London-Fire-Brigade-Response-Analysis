# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pydeck as pdk
import plotly.express as px
import os

st.set_page_config(page_title="London Fire Brigade Response Analysis", layout="wide")

# =======================
# Data Loading Functions
# =======================
@st.cache_data(show_spinner=False)
def try_read_csv(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Failed to load {url}: {e}")
        return None

@st.cache_data(show_spinner=True)
def load_datasets():
    sample_paths = {
        'mobilisation': "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Mobilisation.data.from.January.2009.csv",
        'incident': "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Incident.data.csv",
        'cleaned': "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/cleaned_df.csv"
    }

    mob_df = try_read_csv(sample_paths['mobilisation'])
    inc_df = try_read_csv(sample_paths['incident'])
    clean_df = try_read_csv(sample_paths['cleaned'])
    return mob_df, inc_df, clean_df

# =======================
# Helper Functions
# =======================
def parse_datetime(df, col):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True, dayfirst=True)
        except Exception:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def compute_response_times(mob_df, inc_df):
    df = mob_df.copy()
    df = parse_datetime(df, 'DateAndTimeMobilised')
    df = parse_datetime(df, 'DateAndTimeArrived')

    if 'DateAndTimeMobilised' in df.columns and 'DateAndTimeArrived' in df.columns:
        df['response_seconds'] = (df['DateAndTimeArrived'] - df['DateAndTimeMobilised']).dt.total_seconds()
    else:
        df['response_seconds'] = np.nan

    if 'TravelTimeSeconds' in df.columns:
        df['response_seconds'] = df['response_seconds'].fillna(df['TravelTimeSeconds'])
    if 'AttendanceTimeSeconds' in df.columns:
        df['response_seconds'] = df['response_seconds'].fillna(df['AttendanceTimeSeconds'])

    df.loc[df['response_seconds'] < 0, 'response_seconds'] = np.nan

    if 'IncidentNumber' in df.columns and 'IncidentNumber' in inc_df.columns:
        merged = df.merge(
            inc_df[['IncidentNumber', 'IncGeo_BoroughName', 'Latitude', 'Longitude', 'CalYear', 'HourOfCall']],
            on='IncidentNumber', how='left'
        )
    else:
        merged = df

    merged['CalYear'] = merged['CalYear'].fillna(merged['DateAndTimeMobilised'].dt.year if 'DateAndTimeMobilised' in merged.columns else np.nan)
    return merged

def human_time(seconds):
    if pd.isna(seconds):
        return "NA"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

# =======================
# Load Data
# =======================
st.sidebar.header("📂 Data Loading")
st.sidebar.info("Files auto-loaded from GitHub Releases (v1.0).")
mob_df, inc_df, clean_df = load_datasets()

if mob_df is None or inc_df is None or clean_df is None:
    st.error("❌ Failed to load one or more CSVs. Check your release links or try reloading the page.")
    st.stop()

st.sidebar.success("✅ All datasets loaded successfully!")
st.sidebar.write(f"Mobilisation rows: {mob_df.shape[0]:,}")
st.sidebar.write(f"Incident rows: {inc_df.shape[0]:,}")
st.sidebar.write(f"Cleaned rows: {clean_df.shape[0]:,}")

# =======================
# Merge & Preprocess
# =======================
with st.spinner("Processing and merging data..."):
    merged = compute_response_times(mob_df, inc_df)

if 'DateAndTimeMobilised' in merged.columns:
    merged['year'] = merged['DateAndTimeMobilised'].dt.year
    merged['month'] = merged['DateAndTimeMobilised'].dt.to_period('M').astype(str)
    merged['hour'] = merged['DateAndTimeMobilised'].dt.hour
else:
    merged['year'] = merged['CalYear']
    merged['hour'] = merged['HourOfCall']

borough_col = 'IncGeo_BoroughName' if 'IncGeo_BoroughName' in merged.columns else None

# =======================
# Streamlit Tabs
# =======================
st.title("🚒 London Fire Brigade Response Analysis (Hybrid App)")
tab1, tab2 = st.tabs(["📊 Analysis Dashboard", "🤖 Predict Response Time"])

# =======================
# Tab 1 — Analysis
# =======================
with tab1:
    st.header("Response Time Analysis Dashboard")

    years = sorted(merged['year'].dropna().unique().astype(int))
    boroughs = sorted(merged[borough_col].dropna().unique()) if borough_col else []

    col1, col2 = st.columns(2)
    with col1:
        selected_years = st.multiselect("Select Years", years, default=years[-3:])
    with col2:
        selected_boroughs = st.multiselect("Select Boroughs", boroughs)

    df_view = merged.copy()
    if selected_years:
        df_view = df_view[df_view['year'].isin(selected_years)]
    if selected_boroughs and borough_col:
        df_view = df_view[df_view[borough_col].isin(selected_boroughs)]

    st.subheader("📈 Key Metrics")
    k1, k2, k3 = st.columns(3)
    k1.metric("Average Turnout", human_time(df_view['TurnoutTimeSeconds'].mean()))
    k2.metric("Average Travel", human_time(df_view['TravelTimeSeconds'].mean()))
    k3.metric("Average Response", human_time(df_view['response_seconds'].mean()))

    st.markdown("---")

    if borough_col:
        st.subheader("Average Response Time by Borough")
        borough_agg = df_view.groupby(borough_col)['response_seconds'].mean().reset_index().sort_values('response_seconds')
        fig = px.bar(borough_agg, y=borough_col, x='response_seconds', orientation='h',
                     title="Average Response Time (seconds) by Borough")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend Over Time")
    trend = df_view.groupby('month')['response_seconds'].mean().reset_index()
    fig2 = px.line(trend, x='month', y='response_seconds', title="Monthly Response Time Trend")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Incident Map (Sampled)")
    lat_col = 'Latitude' if 'Latitude' in df_view.columns else 'Incident_Latitude'
    lon_col = 'Longitude' if 'Longitude' in df_view.columns else 'Incident_Longitude'
    if lat_col and lon_col in df_view.columns:
        sample = df_view.dropna(subset=[lat_col, lon_col]).sample(min(2000, len(df_view)), random_state=42)
        st.map(sample[[lat_col, lon_col]])

    st.markdown("---")
    st.dataframe(df_view.head(200))

# =======================
# Tab 2 — Prediction
# =======================
with tab2:
    st.header("Predict Response Time")

    if 'response_seconds' not in merged.columns or merged['response_seconds'].dropna().empty:
        st.warning("No response time data available to train/predict.")
        st.stop()

    numeric_feats = ['hour', 'year', 'PumpOrder'] if 'PumpOrder' in merged.columns else ['hour', 'year']
    cat_feats = [borough_col] if borough_col else []
    df_model = merged.dropna(subset=['response_seconds'] + numeric_feats)

    X = df_model[numeric_feats + cat_feats]
    y = df_model['response_seconds']

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numeric_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])

    model = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=80, random_state=42))
    ])

    train, test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(train, y_train)

    st.success("Model trained successfully!")

    col1, col2, col3 = st.columns(3)
    hour_in = col1.number_input("Hour of Call", min_value=0, max_value=23, value=12)
    year_in = col2.selectbox("Year", years, index=len(years)-1)
    borough_in = col3.selectbox("Borough", boroughs) if borough_col else None

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            'hour': hour_in,
            'year': year_in,
            borough_col: borough_in
        }])
        pred = model.predict(input_data)[0]
        st.metric("Predicted Response Time", f"{pred:.1f} sec", delta=None)
        st.write(f"≈ {human_time(pred)}")

st.sidebar.markdown("---")
st.sidebar.caption("Built & deployed for London Fire Brigade Response Analysis (sha-md).")
