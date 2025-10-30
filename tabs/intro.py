import streamlit as st

sidebar_name = "Introduction"

def run():
    st.title("🚒 London Fire Brigade Analysis")
    st.markdown("""
    This dashboard explores data from the **London Fire Brigade** to identify patterns, 
    trends, and operational insights.  
    Navigate through the tabs to explore:
    - Data Overview
    - Visual Insights
    - Predictive Modelling
    """)
