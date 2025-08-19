import streamlit as st
import os
import streamlit_analytics2
from dotenv import load_dotenv

from tabs import (
    image_search,
    max_diff_simulator,
    optimize_and_simulate,
    quick_start,
    search_claim,
)

load_dotenv()

st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="Claim Advisor – GenAI powered",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

# Add a title and logo
logo, title = st.columns([1, 3])
with logo:
    st.image("claimAdvisor3.png", use_column_width="auto", output_format="PNG")
with title:
    st.title("Claim Advisor – GenAI powered")

tabs = {
    "Quick Start Menu": quick_start,
    "Search Your Claims": search_claim,
    "Search Your Images": image_search,
    "MaxDiff Simulator": max_diff_simulator,
    "Generate & Optimize": optimize_and_simulate,
}

with streamlit_analytics2.track(unsafe_password=os.getenv("ST_APP_ANALYTICS_PP")):
    tab_name = st.sidebar.radio("Select a Tab", list(tabs.keys()))

tabs[tab_name].show()

feedbackUrl = "your_feedback_url_here"  # Replace with your actual feedback URL

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            min-width: 300px !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.sidebar.link_button(label="Report a bug", type="primary", url=feedbackUrl)
