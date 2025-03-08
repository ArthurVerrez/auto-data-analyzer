import streamlit as st
from constants import DEFAULT_SESSION_STATE


def metadata():
    st.set_page_config(
        page_title="Auto Data Analyzer",
        page_icon="⚡",
        initial_sidebar_state="collapsed",
        layout="wide",
    )
    with open("./static/style.css") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Initialize default session state values
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
