from collections import OrderedDict
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ==========================
# CONFIG SECTION
# ==========================

class Member:
    def __init__(self, name: str, linkedin_url: str = None, github_url: str = None) -> None:
        self.name = name
        self.linkedin_url = linkedin_url
        self.github_url = github_url

    def sidebar_markdown(self):
        markdown = f'<b style="display: inline-block; vertical-align: middle; height: 100%">{self.name}</b>'
        if self.linkedin_url is not None:
            markdown += f' <a href={self.linkedin_url} target="_blank"><img src="https://dst-studio-template.s3.eu-west-3.amazonaws.com/linkedin-logo-black.png" alt="linkedin" width="25" style="vertical-align: middle; margin-left: 5px"/></a>'
        if self.github_url is not None:
            markdown += f' <a href={self.github_url} target="_blank"><img src="https://dst-studio-template.s3.eu-west-3.amazonaws.com/github-logo.png" alt="github" width="20" style="vertical-align: middle; margin-left: 5px"/></a>'
        return markdown

TITLE = "London Fire Brigade Analysis Dashboard"
TEAM_MEMBERS = [
    Member(
        name="John Doe",
        linkedin_url="https://www.linkedin.com/in/charlessuttonprofile/",
        github_url="https://github.com/charlessutton",
    ),
    Member("Jane Doe"),
]
PROMOTION = "Promotion Bootcamp Data Scientist - April 2021"

# ==========================
# TAB 1 - INTRO
# ==========================

def intro_tab():
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.title("My Awesome DataScientest Project")
    st.markdown("---")
    st.markdown("""
        This dashboard is built with [Streamlit](https://streamlit.io) as a DataScientest project.
        Explore multiple tabs for insights and visualizations.
    """)

# ==========================
# TAB 2 - SECOND TAB
# ==========================

def second_tab():
    st.title("Second Tab")
    st.markdown("""
        This section demonstrates sample text, charts, and images.
    """)
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=list("ABC"))
    st.line_chart(chart_data)
    st.area_chart(chart_data)

    st.markdown("### Example Image")
    try:
        st.image(Image.open("assets/sample-image.jpg"))
    except:
        st.info("Upload `assets/sample-image.jpg` to see the image here.")

# ==========================
# TAB 3 - THIRD TAB
# ==========================

def third_tab():
    st.title("Third Tab")
    st.markdown("Sample data table:")
    st.dataframe(pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD")))

# ==========================
# STREAMLIT APP LOGIC
# ==========================

st.set_page_config(
    page_title=TITLE,
    page_icon="🚒",
)

st.markdown("""
    <style>
    h1, h2, h3 { color: #d32f2f; }
    </style>
""", unsafe_allow_html=True)

TABS = OrderedDict([
    ("Introduction", intro_tab),
    ("Second Tab", second_tab),
    ("Third Tab", third_tab),
])

def run():
    st.sidebar.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png", width=200)
    tab_name = st.sidebar.radio("Navigation", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {PROMOTION}")
    st.sidebar.markdown("### Team members:")
    for member in TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    # Run selected tab
    TABS[tab_name]()

if __name__ == "__main__":
    run()
