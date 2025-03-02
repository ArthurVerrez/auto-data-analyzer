from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from sections.metadata import metadata
from sections.header import header
from sections.input_form import input_form
from sections.response import response
from sections.footer import footer
from sections.sidebar import sidebar

metadata()
sidebar()
header()
input_form()
if st.session_state["uploaded_file"] and st.session_state["data_description"]:
    response()
footer()
