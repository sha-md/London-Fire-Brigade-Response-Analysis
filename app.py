import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time, random

# ===========================
# APP CONFIG
# ===========================
st.set_page_config(page_title="London Fire Brigade — Response Time Analysis", layout="wide")
st.title("🚒 London Fire Brigade — Response Time Analysis")

DATA_URLS = {
    "mobilisation": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Mobilisation.data.from.January.2009.csv.gz",
    "incident": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Incident.data.csv.gz",
    "cleaned": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/cleaned_df.csv.gz"
}

# ===========================
# SIDEBAR
# ===========================
st.sidebar.header("⚙️ Data Options")
load_full = st.sidebar.checkbox("Load full dataset (slow)", value=False)

# ===========================
# HELPER FUNCTIONS
# ===========================
@st.cache_data(show_spinner=False)
def load_csv_random(url, sample_ratio=0.05):
    """Randomly sample rows from large CSV files across all years."""
    return pd.read_csv(
        url,
        compression="infer",
        low_memory=False,
        skiprows=lambda i: i > 0 and random.random() > sample_ratio
    )

def parse_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, dayfirst=True)
    return df

def compute_response_times(mob_df, inc_df):
    df = mob_df.copy()
    df = parse_datetime(df, "DateAndTimeMobilised")
    df = parse_datetime(df, "DateAndTimeArrived")

    if "DateAndTimeMobilised" in df and "DateAndTimeArrived" in df:
        df["response_seconds"] = (df["DateAndTimeArrived"] - df["DateAndTimeMobilised"]).dt.total_seconds()
    else:
        df["response_seconds"] = np.nan

    for c in ["TravelTimeSeconds", "AttendanceTimeSeconds"]:
        if c in df.columns:
            df["response_seconds"] = df["response_seconds"].fillna(df[c])

    df.loc[df["response_seconds"] < 0, "response_seconds"] = np.nan

    if "IncidentNumber" in df and "IncidentNumber" in inc_df:
        merge_cols = [c for c in ["IncidentNumber", "IncGeo_BoroughName", "Latitude", "Longitude", "CalYear", "HourOfCall"] if c in inc_df.columns]
        merged = df.merge(inc_df[merge_cols], on="IncidentNumber", how="left")
    else:
        merged = df.copy()

    merged["year"] = merged.get("CalYear", pd.NaT)
    if merged["year"].isna().all() and "DateAndTimeMobilised" in merged:
        merged["year"] = merged["DateAndTimeMobilised"].dt.year

    merged["hour"] = merged.get("HourOfCall", np.nan)
    if merged["hour"].isna().all() and "DateAndTimeMobilised" in merged:
        merged["hour"] = merged["DateAndTimeMobilised"].dt.hour

    merged["month"] = (
        merged["DateAndTimeMobilised"].dt.to_period("M").astype(str)
        if "DateAndTimeMobilised" in merged
        else np.nan
    )

    return merged

def human_time(seconds):
    if pd.isna(seconds):
        return "NA"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

# ===========================
# LOAD DATA
# ===========================
with st.spinner("📦 Loading datasets..."):
    mob_df = load_csv_random(DATA_URLS["mobilisation"], sample_ratio=0.05 if not load_full else 1)
    inc_df = load_csv_random(DATA_URLS["incident"], sample_ratio=0.05 if not load_full else 1)
    clean_df = load_csv_random(DATA_URLS["cleaned"], sample_ratio=0.05 if not load_full else 1)

merged = compute_response_times(mob_df, inc_df)
borough_col = "IncGeo_BoroughName" if "IncGeo_BoroughName" in merged else None

st.sidebar.success(f"✅ Data Loaded: {merged.shape[0]:,} rows")

# ===========================
# DASHBOARD TAB
# ===========================
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Predictor"])

with tab1:
    st.subheader("📈 Incident Response Overview")

    years = sorted(merged["year"].dropna().unique().astype(int))
    boroughs = sorted(merged[borough_col].dropna().unique()) if borough_col else []

    c1, c2 = st.columns(2)
    sel_years = c1.multiselect("Select Years", years, default=years[-3:] if len(years) >= 3 else years)
    sel_boroughs = c2.multiselect("Select Boroughs", boroughs)

    dfv = merged.copy()
    if sel_years:
        dfv = dfv[dfv["year"].isin(sel_years)]
    if sel_boroughs:
        dfv = dfv[dfv[borough_col].isin(sel_boroughs)]

    # KPIs
    st.markdown("### 🚒 Key Response Metrics")
    k1, k2, k3 = st.columns(3)
    k1.metric("⏱ Avg Turnout", human_time(dfv["TurnoutTimeSeconds"].mean()))
    k2.metric("🚗 Avg Travel", human_time(dfv["TravelTimeSeconds"].mean()))
    k3.metric("🔥 Avg Total Response", human_time(dfv["response_seconds"].mean()))

    st.markdown("---")

    # Borough-wise average
    if borough_col:
        st.subheader("📍 Borough-wise Average Response Time")
        agg = dfv.groupby(borough_col)["response_seconds"].mean().reset_index().sort_values("response_seconds")
        fig = px.bar(agg, x="response_seconds", y=borough_col, orientation="h",
                     title="Average Response Time by Borough (seconds)", color="response_seconds",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    # Yearly trend
    st.subheader("📆 Yearly Trend in Response Time")
    yearly = dfv.groupby("year")["response_seconds"].mean().reset_index()
    fig2 = px.line(yearly, x="year", y="response_seconds", markers=True, title="Average Response Time Over the Years")
    st.plotly_chart(fig2, use_container_width=True)

    # Hourly pattern
    st.subheader("🕒 Average Response Time by Hour")
    hourly = dfv.groupby("hour")["response_seconds"].mean().reset_index()
    fig3 = px.line(hourly, x="hour", y="response_seconds", markers=True, title="Hourly Response Patterns")
    st.plotly_chart(fig3, use_container_width=True)

# ===========================
# PREDICTOR TAB
# ===========================
with tab2:
    st.header("🤖 Predict Response Time")

    df_model = merged.dropna(subset=["response_seconds", "year", "hour"])
    features = ["hour", "year"]
    if "PumpOrder" in merged:
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

    model.fit(X, y)
    st.success("✅ Model trained on sample data!")

    c1, c2, c3 = st.columns(3)
    hour_in = c1.number_input("Hour of Call", 0, 23, 12)
    all_years = sorted(df_model["year"].dropna().unique().astype(int))
    future_years = all_years + [y for y in range(max(all_years) + 1, 2031)]
    year_in = c2.selectbox("Year", future_years)
    borough_in = c3.selectbox("Borough", sorted(df_model[borough_col].dropna().unique())) if borough_col else None

    if st.button("Predict"):
        inp = pd.DataFrame([{"hour": hour_in, "year": year_in, borough_col: borough_in}])
        for col in X.columns:
            if col not in inp.columns:
                inp[col] = None
        pred = model.predict(inp)[0]
        readable = human_time(pred)

        st.metric("Predicted Response Time", f"{pred:.1f} sec", f"≈ {readable}")
        st.markdown(f"💡 **Interpretation:** For an incident in **{borough_in}** at **{hour_in}:00** during **{year_in}**, "
                    f"the estimated fire crew arrival time is about **{readable}**.")
        if year_in > max(all_years):
            st.info(f"📈 {year_in} is a future year — forecast estimated using historical trends.")

        # --- Forecasted Trend (2009–2030) ---
        st.markdown("---")
        st.subheader("📈 Forecasted Trend (2024–2030)")

        years_future = list(range(min(all_years), 2031))
        sample_hour = 12
        boroughs_list = sorted(df_model[borough_col].dropna().unique()) if borough_col else [None]

        predictions = []
        for yr in years_future:
            year_preds = []
            for br in boroughs_list:
                test_df = pd.DataFrame([{"hour": sample_hour, "year": yr, borough_col: br}])
                for col in X.columns:
                    if col not in test_df.columns:
                        test_df[col] = None
                try:
                    pred_val = model.predict(test_df)[0]
                    year_preds.append(pred_val)
                except Exception:
                    continue
            if year_preds:
                predictions.append({"year": yr, "predicted_response": np.mean(year_preds)})

        forecast_df = pd.DataFrame(predictions)
        cutoff_year = max(all_years)
        forecast_df["type"] = np.where(forecast_df["year"] <= cutoff_year, "Actual", "Forecast")

        fig_forecast = px.line(
            forecast_df,
            x="year",
            y="predicted_response",
            color="type",
            markers=True,
            title="Predicted Average Response Time by Year (Historical vs Forecasted)",
            color_discrete_map={"Actual": "#1f77b4", "Forecast": "#d62728"}
        )
        fig_forecast.update_traces(line=dict(width=3))
        st.plotly_chart(fig_forecast, use_container_width=True)

        last_actual = forecast_df.loc[forecast_df["year"] == cutoff_year, "predicted_response"].values[0]
        last_forecast = forecast_df.loc[forecast_df["year"] == 2030, "predicted_response"].values[0]

        if last_forecast < last_actual:
            st.success(f"🔥 Model suggests response times could improve by {last_actual - last_forecast:.1f} seconds by 2030.")
        else:
            st.warning(f"⚠️ Model suggests response times may worsen by {last_forecast - last_actual:.1f} seconds by 2030.")

st.sidebar.markdown("---")
st.sidebar.caption("📊 Built by shabnam")
