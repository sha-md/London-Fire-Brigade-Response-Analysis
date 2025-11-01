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
import plotly.express as px
import pydeck as pdk
import time

# ===========================
# App Configuration
# ===========================
st.set_page_config(page_title="London Fire Brigade Response Analysis", layout="wide")

SAMPLE_SIZE = 30_000  # reduced for cloud performance

DATA_URLS = {
    "mobilisation": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Mobilisation.data.from.January.2009.csv.gz",
    "incident": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Incident.data.csv.gz",
    "cleaned": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/cleaned_df.csv.gz"
}

# ===========================
# Helper Functions
# ===========================
@st.cache_data(show_spinner=False)
def load_csv(url, nrows=SAMPLE_SIZE):
    """Load a CSV or gzipped CSV directly from GitHub."""
    try:
        df = pd.read_csv(url, compression="infer", low_memory=False, nrows=nrows)
        return df
    except Exception as e:
        st.error(f"❌ Error loading {url}: {e}")
        return pd.DataFrame()

def progress_load_datasets():
    """Load all 3 datasets with progress bar."""
    st.sidebar.write("📦 Loading datasets...")
    progress = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Mobilisation
    status_text.text("Loading Mobilisation data...")
    mob_df = load_csv(DATA_URLS["mobilisation"])
    progress.progress(33)

    # Incident
    status_text.text("Loading Incident data...")
    inc_df = load_csv(DATA_URLS["incident"])
    progress.progress(66)

    # Cleaned
    status_text.text("Loading Cleaned data...")
    clean_df = load_csv(DATA_URLS["cleaned"])
    progress.progress(100)

    time.sleep(0.4)
    status_text.text("✅ All data loaded successfully!")

    return mob_df, inc_df, clean_df

def parse_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, dayfirst=True)
    return df

def compute_response_times(mob_df, inc_df):
    """Compute response time in seconds and merge with incident data."""
    df = mob_df.copy()
    df = parse_datetime(df, "DateAndTimeMobilised")
    df = parse_datetime(df, "DateAndTimeArrived")

    if "DateAndTimeMobilised" in df.columns and "DateAndTimeArrived" in df.columns:
        df["response_seconds"] = (df["DateAndTimeArrived"] - df["DateAndTimeMobilised"]).dt.total_seconds()
    else:
        df["response_seconds"] = np.nan

    for col in ["TravelTimeSeconds", "AttendanceTimeSeconds"]:
        if col in df.columns:
            df["response_seconds"] = df["response_seconds"].fillna(df[col])

    df.loc[df["response_seconds"] < 0, "response_seconds"] = np.nan

    if "IncidentNumber" in df.columns and "IncidentNumber" in inc_df.columns:
        merged = df.merge(
            inc_df[["IncidentNumber", "IncGeo_BoroughName", "Latitude", "Longitude", "CalYear", "HourOfCall"]],
            on="IncidentNumber", how="left"
        )
    else:
        merged = df

    merged["year"] = merged["CalYear"].fillna(
        merged["DateAndTimeMobilised"].dt.year if "DateAndTimeMobilised" in merged.columns else np.nan
    )
    merged["hour"] = (
        merged["DateAndTimeMobilised"].dt.hour
        if "DateAndTimeMobilised" in merged.columns
        else merged.get("HourOfCall")
    )
    merged["month"] = (
        merged["DateAndTimeMobilised"].dt.to_period("M").astype(str)
        if "DateAndTimeMobilised" in merged.columns
        else np.nan
    )

    return merged

def human_time(seconds):
    if pd.isna(seconds):
        return "NA"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

# ===========================
# Load Data
# ===========================
st.sidebar.header("📂 Data Loading")
st.sidebar.info("Auto-loading compressed CSVs from GitHub Release (v1.0)...")

mob_df, inc_df, clean_df = progress_load_datasets()

if mob_df.empty or inc_df.empty or clean_df.empty:
    st.error("❌ One or more datasets failed to load. Please verify release URLs.")
    st.stop()

st.sidebar.success("✅ Data successfully loaded!")
st.sidebar.write(f"Mobilisation rows (sample): {mob_df.shape[0]:,}")
st.sidebar.write(f"Incident rows (sample): {inc_df.shape[0]:,}")
st.sidebar.write(f"Cleaned rows (sample): {clean_df.shape[0]:,}")

# ===========================
# Preprocessing
# ===========================
with st.spinner("Processing and merging datasets..."):
    merged = compute_response_times(mob_df, inc_df)

borough_col = "IncGeo_BoroughName" if "IncGeo_BoroughName" in merged.columns else None

# ===========================
# Layout
# ===========================
st.title("🚒 London Fire Brigade — Response Time Analysis (Cloud Edition)")
tab1, tab2 = st.tabs(["📊 EDA Dashboard", "🤖 Predict Response Time"])

# ========== TAB 1 ==========
with tab1:
    st.header("📈 Analysis Dashboard")

    years = sorted(merged["year"].dropna().unique().astype(int)) if "year" in merged.columns else []
    boroughs = sorted(merged[borough_col].dropna().unique()) if borough_col else []

    c1, c2 = st.columns(2)
    selected_years = c1.multiselect("Select Years", years, default=years[-3:] if len(years) >= 3 else years)
    selected_boroughs = c2.multiselect("Select Boroughs", boroughs)

    df_view = merged.copy()
    if selected_years:
        df_view = df_view[df_view["year"].isin(selected_years)]
    if selected_boroughs and borough_col:
        df_view = df_view[df_view[borough_col].isin(selected_boroughs)]

    # KPIs
    st.subheader("🔥 Key Metrics")
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Turnout", human_time(df_view["TurnoutTimeSeconds"].mean()))
    k2.metric("Avg Travel", human_time(df_view["TravelTimeSeconds"].mean()))
    k3.metric("Avg Response", human_time(df_view["response_seconds"].mean()))

    st.markdown("---")
    if borough_col:
        st.subheader("Response Time by Borough")
        agg = df_view.groupby(borough_col)["response_seconds"].mean().reset_index().sort_values("response_seconds")
        fig = px.bar(agg, x="response_seconds", y=borough_col, orientation="h", title="Average Response Time by Borough (seconds)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend Over Time")
    trend = df_view.groupby("month")["response_seconds"].mean().reset_index()
    fig2 = px.line(trend, x="month", y="response_seconds", title="Average Monthly Response Time")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📍 Map (Sampled Incidents)")
    lat_col = "Latitude" if "Latitude" in df_view.columns else "Incident_Latitude"
    lon_col = "Longitude" if "Longitude" in df_view.columns else "Incident_Longitude"
    if lat_col in df_view.columns and lon_col in df_view.columns:
        map_sample = df_view.dropna(subset=[lat_col, lon_col]).sample(min(1500, len(df_view)), random_state=42)
        st.map(map_sample[[lat_col, lon_col]])

    st.markdown("---")
    st.dataframe(df_view.head(200))

# ========== TAB 2 ==========
with tab2:
    st.header("Predict Response Time")

    if "response_seconds" not in merged or merged["response_seconds"].dropna().empty:
        st.warning("⚠️ Not enough response time data for training.")
        st.stop()

    features = ["hour", "year"]
    if "PumpOrder" in merged.columns:
        features.append("PumpOrder")
    cat_features = [borough_col] if borough_col else []

    df_model = merged.dropna(subset=features + ["response_seconds"])
    X = df_model[features + cat_features]
    y = df_model["response_seconds"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    model = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=80, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    st.success("✅ Model trained on sampled data!")

    c1, c2, c3 = st.columns(3)
    hour_in = c1.number_input("Hour of Call", 0, 23, 12)
    year_in = c2.selectbox("Year", years)
    borough_in = c3.selectbox("Borough", boroughs) if borough_col else None

    if st.button("Predict"):
        inp = pd.DataFrame([{"hour": hour_in, "year": year_in, borough_col: borough_in}])
        pred = model.predict(inp)[0]
        st.metric("Predicted Response Time", f"{pred:.1f} sec")
        st.write(f"≈ {human_time(pred)}")

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Optimized for Streamlit Cloud | Auto-loads data from GitHub Releases | Project by sha-md")
