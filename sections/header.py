import streamlit as st
from constants import CURRENT_VERSION


def header():
    st.markdown(
        f"<h1>⚡ Auto Data Analyzer<small>{CURRENT_VERSION}</small></h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """Welcome! 👋 Just drop a data file and I will analyze it for you ✨"""
    )

    st.divider()
