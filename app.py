# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pydeck as pdk
import plotly.express as px

st.set_page_config(page_title="London Fire Brigade — Response Analysis", layout="wide")

########################################
# Helpers
########################################
@st.cache_data(show_spinner=False)
def try_read_csv(path_or_buffer):
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        try:
            # try with encoding fallback
            return pd.read_csv(path_or_buffer, encoding='latin1')
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return None

def parse_datetime_columns(df, colnames):
    for c in colnames:
        if c in df.columns:
            # try several common formats
            try:
                df[c+'_dt'] = pd.to_datetime(df[c], infer_datetime_format=True, errors='coerce', dayfirst=True)
            except Exception:
                df[c+'_dt'] = pd.to_datetime(df[c], errors='coerce')
    return df

def compute_response_times(mob_df, incident_df, cleaned_df):
    df = mob_df.copy()
    # parse relevant columns
    df = parse_datetime_columns(df, ['DateAndTimeMobilised', 'DateAndTimeMobile', 'DateAndTimeArrived'])
    # compute mobilised->arrived
    if 'DateAndTimeMobilised_dt' in df.columns and 'DateAndTimeArrived_dt' in df.columns:
        df['response_seconds'] = (df['DateAndTimeArrived_dt'] - df['DateAndTimeMobilised_dt']).dt.total_seconds()
    else:
        df['response_seconds'] = np.nan

    # fallback to TravelTimeSeconds or AttendanceTimeSeconds if response missing
    if 'TravelTimeSeconds' in df.columns:
        df['response_seconds'] = df['response_seconds'].fillna(df['TravelTimeSeconds'])
    if 'AttendanceTimeSeconds' in df.columns:
        df['response_seconds'] = df['response_seconds'].fillna(df['AttendanceTimeSeconds'])

    # drop impossible values
    df.loc[df['response_seconds'] < 0, 'response_seconds'] = np.nan

    # combine with incident-level info where possible (join on IncidentNumber)
    if 'IncidentNumber' in df.columns and 'IncidentNumber' in incident_df.columns:
        merged = df.merge(incident_df[['IncidentNumber','IncGeo_BoroughName','Latitude','Longitude','DateOfCall','TimeOfCall']],
                          on='IncidentNumber', how='left', suffixes=('','_inc'))
    else:
        merged = df

    # try to add cleaned_df info too (if has borough etc.)
    if 'IncidentNumber' in merged.columns and 'IncidentNumber' in cleaned_df.columns:
        merged = merged.merge(cleaned_df[['IncidentNumber','IncGeo_BoroughName','Incident_Latitude','Incident_Longitude']],
                              on='IncidentNumber', how='left', suffixes=('','_clean'))
    return merged

def human_seconds(s):
    if pd.isna(s):
        return "NA"
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m}m {sec}s"

########################################
# UI: sidebar - data loading
########################################
st.sidebar.header("Data source / files")
st.sidebar.write("Upload your CSVs or let the app attempt to load sample files from GitHub (raw URLs).")
use_github = st.sidebar.checkbox("Try auto-load from GitHub (sha-md repo)", value=True)

uploaded_mob = st.sidebar.file_uploader("Mobilisation CSV (E LFB Mobilisation data)", type=['csv'])
uploaded_inc = st.sidebar.file_uploader("Incident CSV (E LFB Incident data)", type=['csv'])
uploaded_clean = st.sidebar.file_uploader("cleaned_df.csv (cleaned geography/cats)", type=['csv'])

if use_github and not (uploaded_mob or uploaded_inc or uploaded_clean):
    st.sidebar.write("Attempting to load from GitHub (raw). If your filenames/path differ, upload manually.")
    base_raw = "https://raw.githubusercontent.com/sha-md/London-Fire-Brigade-Response-Analysis/main/"
    sample_paths = {
        'mobilisation': base_raw + "E_LFB_Mobilisation_data.csv",
        'incident': base_raw + "E_LFB_Incident_data.csv",
        'cleaned': base_raw + "cleaned_df.csv"
    }
else:
    sample_paths = {}

########################################
# Load data (uploader or github)
########################################
@st.cache_data(show_spinner=True)
def load_datasets(mob_stream, inc_stream, clean_stream, sample_paths):
    # mobilisation
    if mob_stream is not None:
        mob_df = try_read_csv(mob_stream)
    elif 'mobilisation' in sample_paths:
        mob_df = try_read_csv(sample_paths['mobilisation'])
    else:
        mob_df = None

    if inc_stream is not None:
        inc_df = try_read_csv(inc_stream)
    elif 'incident' in sample_paths:
        inc_df = try_read_csv(sample_paths['incident'])
    else:
        inc_df = None

    if clean_stream is not None:
        clean_df = try_read_csv(clean_stream)
    elif 'cleaned' in sample_paths:
        clean_df = try_read_csv(sample_paths['cleaned'])
    else:
        clean_df = None

    return mob_df, inc_df, clean_df

mob_df, inc_df, clean_df = load_datasets(uploaded_mob, uploaded_inc, uploaded_clean, sample_paths)

if mob_df is None:
    st.warning("Mobilisation dataframe not loaded yet. Upload 'E LFB Mobilisation data' or enable GitHub auto-load.")
    st.stop()

# show basic info
st.sidebar.write("Mobilisation rows:", mob_df.shape[0])
if inc_df is not None:
    st.sidebar.write("Incident rows:", inc_df.shape[0])
if clean_df is not None:
    st.sidebar.write("Cleaned rows:", clean_df.shape[0])

########################################
# Preprocess + enrich
########################################
with st.spinner("Parsing timestamps and computing response times..."):
    merged = compute_response_times(mob_df, inc_df if inc_df is not None else pd.DataFrame(), clean_df if clean_df is not None else pd.DataFrame())

# derive more columns
if 'DateAndTimeMobilised_dt' in merged.columns:
    merged['mobilised_date'] = merged['DateAndTimeMobilised_dt'].dt.date
    merged['mobilised_year'] = merged['DateAndTimeMobilised_dt'].dt.year
    merged['mobilised_month'] = merged['DateAndTimeMobilised_dt'].dt.to_period('M').astype(str)
    merged['mobilised_hour'] = merged['DateAndTimeMobilised_dt'].dt.hour
else:
    # fallback to CalYear/HourOfCall where present
    if 'CalYear' in merged.columns:
        merged['mobilised_year'] = merged['CalYear']
    if 'HourOfCall' in merged.columns:
        merged['mobilised_hour'] = merged['HourOfCall']

# borough column try
borough_col = None
for c in ['IncGeo_BoroughName','IncGeo_BoroughName_clean','IncGeo_BoroughName_cleaned','IncGeo_BoroughName_clean']:
    if c in merged.columns:
        borough_col = c
        break
if borough_col is None:
    # try alternative
    if 'DeployedFromStation_Name' in merged.columns:
        borough_col = 'DeployedFromStation_Name'

########################################
# Main UI layout
########################################
st.title("London Fire Brigade — Response Analysis (Hybrid)")
st.markdown("**Analysis + Prediction** — upload CSVs and explore. This app was generated to run directly (no Jupyter).")

tab1, tab2 = st.tabs(["📊 Analysis Dashboard", "🤖 Predict Response Time"])

########################################
# Dashboard tab
########################################
with tab1:
    st.header("Key metrics & filters")
    col1, col2 = st.columns([2,1])
    with col2:
        st.markdown("**Filters**")
        years = sorted(merged['mobilised_year'].dropna().unique().astype(int).tolist()) if 'mobilised_year' in merged.columns else []
        sel_year = st.multiselect("Year", options=years, default=years if len(years)<=3 else years[-3:])
        boroughs = sorted(merged[borough_col].dropna().unique().tolist()) if borough_col and borough_col in merged.columns else []
        sel_borough = st.multiselect("Borough", options=boroughs, default=None)
        incident_types = []
        for col in ['IncidentGroup','SpecialServiceType','StopCodeDescription','PropertyType']:
            if col in merged.columns:
                incident_types = merged[col].dropna().unique().tolist()
                break
        sel_type = st.selectbox("Incident Type (sample)", options=[None]+(incident_types[:50] if len(incident_types)>0 else []))

    # apply filters
    df_view = merged.copy()
    if sel_year:
        df_view = df_view[df_view['mobilised_year'].isin(sel_year)]
    if sel_borough and borough_col:
        df_view = df_view[df_view[borough_col].isin(sel_borough)]
    if sel_type:
        # try to find first matching column
        for col in ['IncidentGroup','SpecialServiceType','StopCodeDescription','PropertyType']:
            if col in df_view.columns:
                df_view = df_view[df_view[col]==sel_type]
                break

    st.subheader("KPIs")
    k1, k2, k3, k4 = st.columns(4)
    mean_turnout = df_view['TurnoutTimeSeconds'].dropna().mean() if 'TurnoutTimeSeconds' in df_view.columns else np.nan
    mean_travel = df_view['TravelTimeSeconds'].dropna().mean() if 'TravelTimeSeconds' in df_view.columns else np.nan
    mean_attendance = df_view['AttendanceTimeSeconds'].dropna().mean() if 'AttendanceTimeSeconds' in df_view.columns else np.nan
    mean_response = df_view['response_seconds'].dropna().mean()

    k1.metric("Mean Turnout", human_seconds(mean_turnout))
    k2.metric("Mean Travel", human_seconds(mean_travel))
    k3.metric("Mean Attendance", human_seconds(mean_attendance))
    k4.metric("Mean Response", human_seconds(mean_response))

    st.markdown("---")
    st.subheader("Borough-level response time")
    if borough_col and borough_col in df_view.columns:
        borough_agg = df_view.groupby(borough_col)['response_seconds'].agg(['count','median','mean']).reset_index().sort_values('mean', ascending=False)
        fig = px.bar(borough_agg.head(25), x='mean', y=borough_col, orientation='h',
                     labels={'mean':'Mean response (s)', borough_col:'Borough'}, title="Mean response time by borough (top 25)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Borough column not found. Upload the cleaned_df or Incident CSV containing 'IncGeo_BoroughName'.")

    st.markdown("### Geographic map of incidents (sampled for speed)")
    # map plotting: use available lat/lon columns
    lat_col = None
    lon_col = None
    for lat_c in ['Incident_Latitude','Incident_Lat','Latitude','IncidentLatitude','Latitude_inc','Incident_Latitude_clean','Incident_Latitude']:
        if lat_c in df_view.columns:
            lat_col = lat_c
            break
    for lon_c in ['Incident_Longitude','Incident_Long','Longitude','IncidentLongitude','Longitude_inc','Incident_Longitude']:
        if lon_c in df_view.columns:
            lon_col = lon_c
            break

    if lat_col and lon_col:
        sample = df_view.dropna(subset=[lat_col, lon_col, 'response_seconds']).sample(n=min(20000, len(df_view)), random_state=42) if len(df_view)>1000 else df_view.dropna(subset=[lat_col, lon_col, 'response_seconds'])
        sample = sample.copy()
        # normalize response for coloring radius
        sample['resp_clipped'] = sample['response_seconds'].clip(lower=0, upper=3600)
        sample['radius'] = (sample['resp_clipped'] / sample['resp_clipped'].max()) * 1000 + 100
        view = pdk.ViewState(latitude=51.509865, longitude=-0.118092, zoom=9, pitch=0)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=sample[[lat_col, lon_col, 'resp_clipped']].rename(columns={lat_col:'lat', lon_col:'lon'}).to_dict(orient='records'),
            get_position='[lon, lat]',
            get_radius='radius',
            pickable=True,
            auto_highlight=True
        )
        tooltip = {"html": "<b>Response (s):</b> {resp_clipped}", "style": {"color": "white"}}
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))
    else:
        st.info("Latitude/Longitude columns not found in uploaded files. Ensure Incident or cleaned file includes coordinates.")

    st.markdown("---")
    st.subheader("Time series: Response time trend")
    if 'mobilised_month' in df_view.columns and 'response_seconds' in df_view.columns:
        ts = df_view.groupby('mobilised_month')['response_seconds'].median().reset_index()
        fig2 = px.line(ts, x='mobilised_month', y='response_seconds', title="Median response time by month")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Monthly or response time data missing to create trend plot.")

    st.markdown("### Top stations by mean response time")
    if 'DeployedFromStation_Name' in df_view.columns:
        station_agg = df_view.groupby('DeployedFromStation_Name')['response_seconds'].agg(['count','mean']).reset_index().sort_values('mean', ascending=False)
        st.dataframe(station_agg.head(20).assign(mean=lambda d: d['mean'].round(1)))
    else:
        st.info("Station column not available.")

    st.markdown("---")
    st.subheader("Raw / sample data (for quick view)")
    st.dataframe(df_view.sample(n=min(200, len(df_view))).reset_index(drop=True))

########################################
# Prediction tab
########################################
with tab2:
    st.header("Predict response time")
    st.markdown("This tab will try to load a saved scaler + model from `scaler.joblib` and `models/` folder. If not found, you can train a quick baseline model here (takes a minute).")

    # try to load scaler and model from repo (relative paths)
    loaded_scaler = None
    loaded_model = None
    try_paths = [
        "scaler.joblib",
        "models/scaler.joblib",
        "models/scaler.pkl",
        "models/model.joblib",
        "models/model.pkl",
        "models/rf_model.joblib",
        "models/xgb_model.joblib"
    ]
    for p in try_paths:
        if os.path.exists(p):
            try:
                obj = joblib.load(p)
                # heuristics: scaler vs model
                if hasattr(obj, 'transform') and loaded_scaler is None:
                    loaded_scaler = obj
                else:
                    loaded_model = obj
            except Exception:
                pass

    if loaded_scaler is None:
        # also try joblib load from root scaler.joblib (repo)
        try:
            loaded_scaler = joblib.load('scaler.joblib')
        except Exception:
            loaded_scaler = None

    if loaded_model is None:
        # try model in models/
        for fname in os.listdir("models") if os.path.isdir("models") else []:
            try:
                obj = joblib.load(os.path.join("models", fname))
                if not hasattr(obj, 'transform'):
                    loaded_model = obj
                    break
            except Exception:
                pass

    if loaded_model is not None:
        st.success("Loaded pre-trained model from repo.")
    else:
        st.info("No pre-trained model found in repo.")

    # Feature selection for model
    feature_cols = []
    candidate_features = ['mobilised_hour','CalYear','PumpOrder','PumpCount','NumPumpsAttending']
    for f in candidate_features:
        if f in merged.columns:
            feature_cols.append(f)
    # Borough as categorical
    borough_feature = borough_col if borough_col and borough_col in merged.columns else None

    st.subheader("Model inputs")
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        hour_in = st.number_input("Hour of call (0-23)", min_value=0, max_value=23, value=12)
        year_in = st.selectbox("Year", options=sorted(merged['mobilised_year'].dropna().unique().astype(int).tolist()) if 'mobilised_year' in merged.columns else [2019], index=0)
        pump_order_in = st.number_input("PumpOrder (if available)", min_value=0, max_value=10, value=1)
    with input_col2:
        borough_in = st.selectbox("Borough (optional)", options=[None]+(boroughs if boroughs else []))
        num_pumps = st.number_input("NumPumpsAttending (if known)", min_value=0, max_value=20, value=1)

    # button to predict
    do_predict = st.button("Predict response time")

    # If no model, allow training a quick baseline
    if loaded_model is None:
        train_quick = st.checkbox("Train a quick baseline RandomForest (on available rows) — caches result", value=False)
        rf_model = None
        pipeline = None
        if train_quick:
            st.info("Training baseline model: this uses available rows with non-null response and basic features (may take ~30-90s depending on dataset).")
            # prepare dataset
            df_model = merged.copy()
            # ensure features exist
            # create borough as category if present
            use_cols = []
            if 'mobilised_hour' in df_model.columns:
                use_cols.append('mobilised_hour')
            if 'mobilised_year' in df_model.columns:
                use_cols.append('mobilised_year')
            if 'PumpOrder' in df_model.columns:
                use_cols.append('PumpOrder')
            if 'NumPumpsAttending' in df_model.columns:
                use_cols.append('NumPumpsAttending')
            if borough_feature:
                use_cols.append(borough_feature)

            df_model = df_model.dropna(subset=use_cols + ['response_seconds'])
            if len(df_model) < 200:
                st.warning("Not enough rows to train a reliable model. Need at least ~200 rows with the required columns.")
            else:
                X = df_model[use_cols]
                y = df_model['response_seconds']
                # build preprocessing pipeline
                numeric_feats = [c for c in use_cols if c != borough_feature]
                cat_feats = [borough_feature] if borough_feature in use_cols else []
                preprocessor = ColumnTransformer([
                    ('num', 'passthrough', numeric_feats),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
                ])
                pipeline = Pipeline([
                    ('pre', preprocessor),
                    ('rf', RandomForestRegressor(n_estimators=80, max_depth=10, n_jobs=-1, random_state=42))
                ])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                pipeline.fit(X_train, y_train)
                rf_model = pipeline
                st.success("Quick baseline trained and cached.")

    ####################
    # Perform prediction
    ####################
    if do_predict:
        # prepare input row
        input_df = pd.DataFrame([{
            'mobilised_hour': hour_in,
            'mobilised_year': year_in,
            'PumpOrder': pump_order_in,
            'NumPumpsAttending': num_pumps,
            borough_feature: borough_in if borough_feature else None
        }])
        # align with model
        if loaded_model is not None:
            try:
                # if loaded_model is a pipeline or has predict
                pred = loaded_model.predict(input_df.select_dtypes(include=[np.number, object], errors='ignore'))
                st.metric("Predicted response (s)", f"{pred[0]:.1f}", delta=None)
                st.write("Estimated:", human_seconds(pred[0]))
            except Exception as e:
                # attempt to use pipeline-style predict (if preprocessor missing)
                try:
                    pred = loaded_model.predict(input_df)
                    st.metric("Predicted response (s)", f"{pred[0]:.1f}")
                    st.write("Estimated:", human_seconds(pred[0]))
                except Exception as e2:
                    st.error(f"Model present but prediction failed: {e}; {e2}")
        elif rf_model is not None:
            try:
                # ensure columns match
                pred = rf_model.predict(input_df)
                st.metric("Predicted response (s)", f"{pred[0]:.1f}")
                st.write("Estimated:", human_seconds(pred[0]))
            except Exception as e:
                st.error(f"Prediction error with trained baseline: {e}")
        else:
            st.warning("No model available. Train a quick baseline or add a saved model into the repo (scaler.joblib, models/*.joblib).")

    st.markdown("---")
    st.markdown("**Model files:** You can add `scaler.joblib` and a pipeline/model file into the repo `models/` folder. The app will try to auto-load them on startup.")

########################################
# Footer
########################################
st.sidebar.markdown("---")
st.sidebar.write("Streamlit app generated for your London Fire Brigade project.")
st.sidebar.write("Notes:")
st.sidebar.markdown("""
- If your filenames are different in the repo, upload them manually.  
- To deploy on Streamlit Cloud: push this `app.py` and the data/model files to GitHub and then deploy.  
- If dataset is very large, the app samples for plotting to keep UI responsive.  
""")
