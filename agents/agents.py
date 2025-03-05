#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from agents.crew import AnalystCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


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
    }

    try:
        return (
            AnalystCrew(llm_id)
            .crew(step_callback=step_callback, task_callback=task_callback)
            .kickoff(inputs=inputs)
        )
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
