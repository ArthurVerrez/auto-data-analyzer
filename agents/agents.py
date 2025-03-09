#!/usr/bin/env python
import warnings
import streamlit as st

from constants import MAX_POINTS_BAR_CHART, MAX_POINTS_LINE_CHART

from agents.crew import AnalystCrew
import json

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


@st.cache_data
def run(
    data_description_prompt,
    n_bar_chart_visualizations,
    n_line_chart_visualizations,
    llm_id,
    step_callback=None,
    task_callback=None,
):
    """
    Run the crew.
    """
    inputs = {
        "data_description_prompt": data_description_prompt,
        "n_bar_chart_visualizations": n_bar_chart_visualizations,
        "n_line_chart_visualizations": n_line_chart_visualizations,
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


def get_chart_suggestions(
    data_description_prompt,
    n_bar_chart_visualizations,
    n_line_chart_visualizations,
    llm_id,
):
    response = run(
        data_description_prompt=data_description_prompt,
        n_bar_chart_visualizations=n_bar_chart_visualizations,
        n_line_chart_visualizations=n_line_chart_visualizations,
        llm_id=llm_id,
        step_callback=None,
        task_callback=None,
    )
    bar_chart_suggestions = []
    line_chart_suggestions = []

    for task_output in response.tasks_output:
        if task_output.name == "bar_chart_suggestion_task":
            bar_chart_suggestions = json.loads(task_output.raw)
        elif task_output.name == "line_chart_suggestion_task":
            line_chart_suggestions = json.loads(task_output.raw)

    return {
        "bar_charts": bar_chart_suggestions,
        "line_charts": line_chart_suggestions,
    }
