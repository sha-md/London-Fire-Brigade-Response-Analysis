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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sidebar_name = "Visualizations"

def run():
    st.header("📈 Visual Analysis")
    df = pd.read_csv("fire_brigade_data.csv")

    fig, ax = plt.subplots()
    sns.countplot(x="IncidentGroup", data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write("You can add more visuals here — like incidents by borough, time of day, etc.")
