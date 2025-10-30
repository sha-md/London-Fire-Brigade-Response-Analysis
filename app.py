from collections import OrderedDict
import streamlit as st
import config
from tabs import intro, data_exploration, visualization, modelling

st.set_page_config(
    page_title=config.TITLE,
    page_icon="🚒",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

TABS = OrderedDict([
    (intro.sidebar_name, intro),
    (data_exploration.sidebar_name, data_exploration),
    (visualization.sidebar_name, visualization),
    (modelling.sidebar_name, modelling),
])

def run():
    st.sidebar.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png", width=200)
    tab_name = st.sidebar.radio("Navigation", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")
    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)
    tab = TABS[tab_name]
    tab.run()

if __name__ == "__main__":
    run()
