import os
import streamlit as st
from constants import LLM_OPTIONS


def input_form():
    form = st.form("agent_form")
    st.session_state.api_key = form.text_input(
        "OpenAI API Key _(only stored during this session)_",
        type="password",
        value=os.getenv("LLM_API_KEY") or "",
    )
    st.session_state.data_description = form.text_input(
        "Describe your data",
        placeholder="A list of transactions with date, amount and other details about the merchants",
    )

    st.session_state.uploaded_file = form.file_uploader(
        "Upload a data file",
        type=["csv", "xlsx", "xls", "parquet"],
        accept_multiple_files=False,
    )

    st.session_state.llm_id = form.pills(
        "Model",
        options=LLM_OPTIONS.keys(),
        format_func=lambda option: LLM_OPTIONS[option],
        selection_mode="single",
        default=next(iter(LLM_OPTIONS.keys())),
    )

    form.form_submit_button("Submit", type="primary")
