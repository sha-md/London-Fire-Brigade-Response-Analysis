import streamlit as st
import pandas as pd

sidebar_name = "Data Exploration"

def run():
    st.header("📊 Data Overview")
    df = pd.read_csv("fire_brigade_data.csv")

    st.write("#### Dataset Preview")
    st.dataframe(df.head())

    st.write("#### Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Missing values:")
    st.dataframe(df.isna().sum())
