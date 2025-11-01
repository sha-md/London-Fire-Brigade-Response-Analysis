# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time

# ===========================
# App Configuration
# ===========================
st.set_page_config(page_title="London Fire Brigade Response Analysis", layout="wide")

SAMPLE_SIZE = 30000  # sample rows for performance
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
    """Load a CSV or gzipped CSV safely from GitHub."""
    return pd.read_csv(url, compression="infer", low_memory=False, nrows=nrows)

def progress_load():
    """Display progress bar while loading all data."""
    st.sidebar.write("📦 Loading datasets...")
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()

    status.text("Loading Mobilisation data...")
    mob = load_csv(DATA_URLS["mobilisation"])
    progress.progress(33)

    status.text("Loading Incident data...")
    inc = load_csv(DATA_URLS["incident"])
    progress.progress(66)

    status.text("Loading Cleaned data...")
    clean = load_csv(DATA_URLS["cleaned"])
    progress.progress(100)

    time.sleep(0.5)
    status.text("✅ All data loaded successfully!")
    return mob, inc, clean

def parse_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, dayfirst=True)
    return df

def compute_response_times(mob_df, inc_df):
    df = mob_df.copy()
    df = parse_datetime(df, "DateAndTimeMobilised")
    df = parse_datetime(df, "DateAndTimeArrived")

    if "DateAndTimeMobilised" in df.columns and "DateAndTimeArrived" in df.columns:
        df["response_seconds"] = (df["DateAndTimeArrived"] - df["DateAndTimeMobilised"]).dt.total_seconds()
    else:
        df["response_seconds"] = np.nan

    for c in ["TravelTimeSeconds", "AttendanceTimeSeconds"]:
        if c in df.columns:
            df["response_seconds"] = df["response_seconds"].fillna(df[c])

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

mob_df, inc_df, clean_df = progress_load()

if mob_df.empty or inc_df.empty or clean_df.empty:
    st.error("❌ One or more datasets failed to load. Check release URLs.")
    st.stop()

st.sidebar.success("✅ Datasets loaded successfully!")
st.sidebar.write(f"Mobilisation rows (sample): {mob_df.shape[0]:,}")
st.sidebar.write(f"Incident rows (sample): {inc_df.shape[0]:,}")
st.sidebar.write(f"Cleaned rows (sample): {clean_df.shape[0]:,}")

with st.spinner("Processing datasets..."):
    merged = compute_response_times(mob_df, inc_df)

borough_col = "IncGeo_BoroughName" if "IncGeo_BoroughName" in merged.columns else None

# ===========================
# Dashboard Tabs
# ===========================
st.title("🚒 London Fire Brigade – Response Time Analysis")
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Predictor"])

# ---- TAB 1: Dashboard ----
with tab1:
    st.header("📈 Interactive Analysis")

    years = sorted(merged["year"].dropna().unique().astype(int)) if "year" in merged.columns else []
    boroughs = sorted(merged[borough_col].dropna().unique()) if borough_col else []

    c1, c2 = st.columns(2)
    sel_years = c1.multiselect("Select Years", years, default=years[-3:] if len(years) >= 3 else years)
    sel_boroughs = c2.multiselect("Select Boroughs", boroughs)

    dfv = merged.copy()
    if sel_years:
        dfv = dfv[dfv["year"].isin(sel_years)]
    if sel_boroughs and borough_col:
        dfv = dfv[dfv[borough_col].isin(sel_boroughs)]

    # KPIs
    st.subheader("🔥 Key Metrics")
    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Turnout", human_time(dfv["TurnoutTimeSeconds"].mean()))
    k2.metric("Avg Travel", human_time(dfv["TravelTimeSeconds"].mean()))
    k3.metric("Avg Response", human_time(dfv["response_seconds"].mean()))

    st.markdown("---")
    if borough_col:
        st.subheader("Response Time by Borough")
        agg = dfv.groupby(borough_col)["response_seconds"].mean().reset_index().sort_values("response_seconds")
        fig = px.bar(agg, x="response_seconds", y=borough_col, orientation="h",
                     title="Average Response Time (seconds) by Borough")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend Over Time")
    trend = dfv.groupby("month")["response_seconds"].mean().reset_index()
    fig2 = px.line(trend, x="month", y="response_seconds", title="Average Monthly Response Time")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📍 Map of Incidents (sample)")
    lat = "Latitude" if "Latitude" in dfv.columns else "Incident_Latitude"
    lon = "Longitude" if "Longitude" in dfv.columns else "Incident_Longitude"
    if lat in dfv.columns and lon in dfv.columns:
        smp = dfv.dropna(subset=[lat, lon]).sample(min(1500, len(dfv)), random_state=42)
        st.map(smp[[lat, lon]])

    st.markdown("---")
    st.dataframe(dfv.head(200))

# ---- TAB 2: Predictor ----
with tab2:
    st.header("Predict Response Time (Simple Model)")
    if "response_seconds" not in merged or merged["response_seconds"].dropna().empty:
        st.warning("⚠️ Not enough data for training.")
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
    st.success("✅ Model trained successfully on sample data!")

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
st.sidebar.caption("⚡ Cloud-Optimized Streamlit App | Data: GitHub Releases | Project by sha-md")
