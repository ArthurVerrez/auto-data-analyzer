# The next 3 lines are here for compatibility with the Streamlit Cloud platform
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from sections.metadata import metadata
from sections.header import header
from sections.input_form import input_form
from sections.response import response
from sections.footer import footer
from sections.sidebar import sidebar

from utils import get_df_from_file

metadata()
sidebar()
header()
input_form()

if st.session_state.uploaded_file:
    st.session_state.df = get_df_from_file(st.session_state.uploaded_file)

if st.session_state.df is not None and st.session_state.data_description:
    response()

footer()
