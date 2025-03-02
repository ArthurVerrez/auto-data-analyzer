import streamlit as st
import pandas as pd


def response():
    df = st.session_state.df

    st.write(df)
