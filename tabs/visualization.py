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
