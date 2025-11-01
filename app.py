# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="London Fire Brigade – Lightweight Dashboard", layout="wide")

# --- CONFIG ---
DATA_URLS = {
    "mobilisation": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Mobilisation.data.from.January.2009.csv.gz",
    "incident": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/LFB.Incident.data.csv.gz",
    "cleaned": "https://github.com/sha-md/London-Fire-Brigade-Response-Analysis/releases/download/v1.0/cleaned_df.csv.gz"
}
SAMPLE_SIZE = 5_000  # safe default for cloud


# --- HELPERS ---
@st.cache_data(show_spinner=False)
def load_csv(url, nrows=SAMPLE_SIZE):
    """Load a small sample directly from a compressed CSV."""
    return pd.read_csv(url, compression="infer", low_memory=False, nrows=nrows)


@st.cache_data(show_spinner=False)
def compute_summary(df):
    """Very light summary so app stays fast."""
    return {
        "rows": len(df),
        "columns": list(df.columns[:10]),
        "avg_response": float(df.get("AttendanceTimeSeconds", pd.Series([np.nan])).mean())
    }


# --- APP BODY ---
st.title("🚒 London Fire Brigade – Response Dashboard (Lazy Load)")

st.markdown("""
This lightweight version waits until you click **Load Data**.  
That avoids timeouts on Streamlit Cloud while still showing your visualisations.
""")

if st.button("🔽 Load 5 000-row Sample Now"):
    with st.spinner("Fetching small samples from GitHub…"):
        mob = load_csv(DATA_URLS["mobilisation"])
        inc = load_csv(DATA_URLS["incident"])
        clean = load_csv(DATA_URLS["cleaned"])

    st.success("✅ Data loaded successfully!")
    st.write("**Mobilisation sample:**", compute_summary(mob))
    st.write("**Incident sample:**", compute_summary(inc))
    st.write("**Cleaned sample:**", compute_summary(clean))

    if "IncGeo_BoroughName" in inc.columns and "AttendanceTimeSeconds" in inc.columns:
        borough_avg = (
            inc.groupby("IncGeo_BoroughName")["AttendanceTimeSeconds"]
            .mean()
            .reset_index()
            .sort_values("AttendanceTimeSeconds")
        )
        fig = px.bar(
            borough_avg,
            y="IncGeo_BoroughName",
            x="AttendanceTimeSeconds",
            orientation="h",
            title="Average Attendance Time by Borough (sample)"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click **Load Data** above to fetch small samples from GitHub Releases.")

st.caption("Optimised for Streamlit Cloud free tier – sha-md project.")
