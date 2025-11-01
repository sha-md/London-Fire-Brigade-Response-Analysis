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

st.title("🚒 London Fire Brigade — Response Time Analysis")

DATA_URLS = {
    "mobilisation": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Mobilisation.data.from.January.2009.csv.gz",
    "incident": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Incident.data.csv.gz",
    "cleaned": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/cleaned_df.csv.gz"
}

# ===========================
# Sidebar Controls
# ===========================
st.sidebar.header("📂 Data Loading Options")

load_full = st.sidebar.checkbox("Load full dataset (may take longer)", value=False)
SAMPLE_SIZE = None if load_full else 200_000  # load more rows for more years

st.sidebar.info("Auto-loading compressed CSVs from GitHub Release (v1.0)...")

# ===========================
# Helper Functions
# ===========================
@st.cache_data(show_spinner=False)
def load_csv(url, nrows=SAMPLE_SIZE, usecols=None):
    """Load CSV or gzipped CSV from GitHub with optional column selection."""
    return pd.read_csv(url, compression="infer", low_memory=False, nrows=nrows, usecols=usecols)

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
        merge_cols = [c for c in ["IncidentNumber", "IncGeo_BoroughName", "Latitude", "Longitude", "CalYear", "HourOfCall"] if c in inc_df.columns]
        merged = df.merge(inc_df[merge_cols], on="IncidentNumber", how="left")
    else:
        merged = df.copy()

    # Add fallback year/hour/month
    if "CalYear" in merged.columns:
        merged["year"] = merged["CalYear"]
    elif "DateAndTimeMobilised" in merged.columns:
        merged["year"] = merged["DateAndTimeMobilised"].dt.year
    else:
        merged["year"] = np.nan

    if "HourOfCall" in merged.columns:
        merged["hour"] = merged["HourOfCall"]
    elif "DateAndTimeMobilised" in merged.columns:
        merged["hour"] = merged["DateAndTimeMobilised"].dt.hour
    else:
        merged["hour"] = np.nan

    if "DateAndTimeMobilised" in merged.columns:
        merged["month"] = merged["DateAndTimeMobilised"].dt.to_period("M").astype(str)
    else:
        merged["month"] = np.nan

    return merged

def human_time(seconds):
    if pd.isna(seconds):
        return "NA"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

# ===========================
# Load Data
# ===========================
mob_df, inc_df, clean_df = progress_load()

if mob_df.empty or inc_df.empty or clean_df.empty:
    st.error("❌ One or more datasets failed to load.")
    st.stop()

st.sidebar.success("✅ Datasets loaded successfully!")

with st.spinner("Processing datasets..."):
    merged = compute_response_times(mob_df, inc_df)

borough_col = "IncGeo_BoroughName" if "IncGeo_BoroughName" in merged.columns else None

# ===========================
# Tabs Layout
# ===========================
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Predictor"])

# ---- TAB 1 ----
with tab1:
    st.header("📈 Interactive Analysis")

    years = sorted(merged["year"].dropna().unique().astype(int))
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

    # Charts
    st.markdown("---")
    if borough_col:
        st.subheader("Response Time by Borough")
        agg = dfv.groupby(borough_col)["response_seconds"].mean().reset_index().sort_values("response_seconds")
        fig = px.bar(agg, x="response_seconds", y=borough_col, orientation="h", title="Average Response Time (seconds) by Borough")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend Over Time")
    trend = dfv.groupby("month")["response_seconds"].mean().reset_index()
    fig2 = px.line(trend, x="month", y="response_seconds", title="Average Monthly Response Time")
    st.plotly_chart(fig2, use_container_width=True)

    # Map
    st.subheader("📍 Map of Incidents (sample)")
    possible_lat = ["Latitude", "Incident_Latitude", "Incident_Lat", "lat"]
    possible_lon = ["Longitude", "Incident_Longitude", "Incident_Lon", "lon"]
    lat_col = next((c for c in possible_lat if c in dfv.columns), None)
    lon_col = next((c for c in possible_lon if c in dfv.columns), None)

    if lat_col and lon_col:
        smp = dfv.dropna(subset=[lat_col, lon_col]).sample(min(1500, len(dfv)), random_state=42)
        smp = smp.rename(columns={lat_col: "latitude", lon_col: "longitude"})
        st.map(smp[["latitude", "longitude"]])
    else:
        st.warning("⚠️ Latitude/Longitude columns not found, skipping map plot.")

# ---- TAB 2 ----
with tab2:
    st.header("Predict Response Time (Simple Model)")
    df_model = merged.dropna(subset=["response_seconds", "year", "hour"])
    features = ["hour", "year"]
    if "PumpOrder" in merged.columns:
        features.append("PumpOrder")
    cat_features = [borough_col] if borough_col else []

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
    # include future years up to 2030
    all_years = sorted(df_model["year"].dropna().unique().astype(int))
    extended_years = all_years + [y for y in range(max(all_years) + 1, 2031)]
    year_in = c2.selectbox("Year", extended_years)
    borough_in = c3.selectbox("Borough", sorted(df_model[borough_col].dropna().unique())) if borough_col else None

    if st.button("Predict"):
        input_data = {"hour": hour_in, "year": year_in}
        if "PumpOrder" in merged.columns:
            input_data["PumpOrder"] = merged["PumpOrder"].median()
        if borough_col:
            input_data[borough_col] = borough_in

        inp = pd.DataFrame([input_data])
        for col in X.columns:
            if col not in inp.columns:
                inp[col] = None

        pred = model.predict(inp)[0]
        readable = human_time(pred)
        st.metric("Predicted Response Time", f"{pred:.1f} sec")
        st.write(f"≈ {readable}")

        # Interpretation box
        st.markdown(f"""
        💡 **Interpretation:**
        For an incident in **{borough_in}** at **{hour_in}:00 hours** during **{year_in}**,  
        the estimated average fire brigade arrival time is about **{readable}**.
        """)

        if year_in > max(all_years):
            st.info(f"📈 Note: {year_in} is a future year. The prediction extrapolates using past patterns (up to {max(all_years)}).")

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Streamlit App | Data: GitHub Releases | Built by sha-md")
