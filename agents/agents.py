#!/usr/bin/env python
import warnings
import streamlit as st

from constants import MAX_POINTS_BAR_CHART, MAX_POINTS_LINE_CHART

from agents.crew import AnalystCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


@st.cache_data
def run(
    data_description_prompt,
    n_visualizations,
    llm_id,
    step_callback=None,
    task_callback=None,
):
    """
    Run the crew.
    """
    inputs = {
        "data_description_prompt": data_description_prompt,
        "n_visualizations": n_visualizations,
        "max_points_bar_chart": MAX_POINTS_BAR_CHART,
        "max_points_line_chart": MAX_POINTS_LINE_CHART,
    }

    try:
        return (
            AnalystCrew(llm_id)
            .crew(step_callback=step_callback, task_callback=task_callback)
            .kickoff(inputs=inputs)
        )
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
