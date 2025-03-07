#!/usr/bin/env python
import sys
import warnings
import streamlit as st
import json

from datetime import datetime

from agents.crew import AnalystCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


@st.cache_data
def run(
    data_description_prompt,
    n_visualizations,
    llm_id,
    chart_type,
    step_callback=None,
    task_callback=None,
):
    """
    Run the crew.
    """
    inputs = {
        "data_description_prompt": data_description_prompt,
        "n_visualizations": n_visualizations,
    }

    try:
        return (
            AnalystCrew(llm_id, chart_type)
            .crew(step_callback=step_callback, task_callback=task_callback)
            .kickoff(inputs=inputs)
        )
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
