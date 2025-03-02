import streamlit as st
import pandas as pd


def response():
    st.write(st.session_state["data_description"])
    df = pd.read_csv(st.session_state["uploaded_file"])
    st.write(df)
