from numpy import histogram
import streamlit as st
import pandas as pd
from utils import (
    get_column_statistics,
    bar_chart_data_generator,
    timestamp_converter,
    line_chart_data_generator,
    histogram_data_generator,
)
from agents.agents import get_chart_suggestions
import os
import json
import plotly.express as px


MAX_DISTINCT = 10000  # maximum allowed distinct values for charting


def response() -> None:
    """Display raw data, column summaries with statistics, and random full rows.

    Numeric columns: min, max, mean, median, std.
    Datetime columns: min, max.
    Boolean columns: counts of True and False.
    Other columns: number of distinct values and mode.
    """

    # Set the OPENAI_API_KEY environment variable
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key

    df = st.session_state.df
    col_stats = get_column_statistics(df)
    example_rows = df.sample(n=min(5, len(df)), random_state=42)

    st.divider()

    with st.expander("See the raw data", expanded=False):
        st.dataframe(df, use_container_width=True)

    with st.expander("Column Summaries", expanded=False):
        st.write(col_stats)

    with st.expander("Example rows", expanded=False):
        # Show as a csv
        st.write(example_rows.to_csv(index=False))

    data_description_prompt = f"""
    The user has uploaded the file {st.session_state.uploaded_file.name} and described it as follows:
    ```
    {st.session_state.data_description}
    ```

    Here are a few randomly chosen examples of the data:
    ```
    {example_rows.to_csv(index=False)}
    ```


    Here are some advanced insights into the data by columns:
    ```json
    {json.dumps(col_stats, indent=4, default=timestamp_converter)}
    ```
    """

    with st.expander("Data Description Prompt", expanded=False):
        st.write(data_description_prompt)

    st.divider()

    with st.spinner("Getting chart suggestions..."):
        chart_suggestions = get_chart_suggestions(
            data_description_prompt=data_description_prompt,
            n_bar_chart_visualizations=5,
            n_line_chart_visualizations=5,
            n_histogram_visualizations=5,
            llm_id=st.session_state.llm_id,
        )
        bar_chart_suggestions = chart_suggestions.get("bar_charts", [])
        line_chart_suggestions = chart_suggestions.get("line_charts", [])
        histogram_suggestions = chart_suggestions.get("histograms", [])

    with st.expander("Bar Chart Suggestions", expanded=False):
        st.write(bar_chart_suggestions)

    with st.expander("Line Chart Suggestions", expanded=False):
        st.write(line_chart_suggestions)

    with st.expander("Histogram Suggestions", expanded=False):
        st.write(histogram_suggestions)

    rows, cols = (len(bar_chart_suggestions) + 2) // 3, 3

    for row in range(rows):
        columns = st.columns(cols)
        for col_idx, col in enumerate(columns):
            chart_index = row * cols + col_idx
            if chart_index < len(bar_chart_suggestions):
                chart_config = bar_chart_suggestions[chart_index]
                with col:
                    try:
                        st.write(f"### {chart_config['title']}")
                        st.plotly_chart(
                            px.bar(**bar_chart_data_generator(df, chart_config))
                        )
                    except Exception as e:
                        st.write(
                            f"Error: {e} occurred while rendering the chart. Please try another chart suggestion."
                        )

    rows, cols = (len(line_chart_suggestions) + 2) // 3, 3

    for row in range(rows):
        columns = st.columns(cols)
        for col_idx, col in enumerate(columns):
            chart_index = row * cols + col_idx
            if chart_index < len(line_chart_suggestions):
                chart_config = line_chart_suggestions[chart_index]
                with col:
                    try:
                        st.write(f"### {chart_config['title']}")
                        st.plotly_chart(
                            px.line(**line_chart_data_generator(df, chart_config))
                        )
                    except Exception as e:
                        st.write(
                            f"Error: {e} occurred while rendering the chart. Please try another chart suggestion."
                        )

    rows, cols = (len(histogram_suggestions) + 2) // 3, 3

    for row in range(rows):
        columns = st.columns(cols)
        for col_idx, col in enumerate(columns):
            chart_index = row * cols + col_idx
            if chart_index < len(histogram_suggestions):
                chart_config = histogram_suggestions[chart_index]
                with col:
                    try:
                        st.write(f"### {chart_config['title']}")
                        st.plotly_chart(
                            px.histogram(**histogram_data_generator(df, chart_config))
                        )
                    except Exception as e:
                        st.write(
                            f"Error: {e} occurred while rendering the chart. Please try another chart suggestion."
                        )
