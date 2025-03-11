import os
import json
from typing import Any, Callable, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from utils import (
    get_column_statistics,
    bar_chart_data_generator,
    timestamp_converter,
    line_chart_data_generator,
    histogram_data_generator,
)
from agents.agents import get_chart_suggestions

MAX_DISTINCT = 10000  # maximum allowed distinct values for charting


def render_charts(
    suggestions: List[Dict[str, Any]],
    df: pd.DataFrame,
    generator: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
    chart_func: Callable[..., Any],
    chart_type: str,
) -> None:
    """Render charts based on suggestions in a grid layout.

    Args:
        suggestions: List of chart configuration dictionaries.
        df: DataFrame with the data.
        generator: Function to generate chart data.
        chart_func: Plotly chart function (e.g., px.bar, px.line, px.histogram).
        chart_type: A string indicating the chart type (used for error messages).
    """
    cols = 3
    rows = (len(suggestions) + cols - 1) // cols
    for row in range(rows):
        columns = st.columns(cols)
        for col_idx, col in enumerate(columns):
            chart_index = row * cols + col_idx
            if chart_index < len(suggestions):
                chart_config = suggestions[chart_index]
                with col:
                    try:
                        chart_data = generator(df, chart_config)
                        chart = chart_func(**chart_data)
                        st.plotly_chart(chart)
                    except Exception as e:
                        st.error(
                            f"Error: {e} occurred while rendering the {chart_type} chart. Please try another chart suggestion."
                        )


def response() -> None:
    """Display raw data, column summaries with statistics, and random full rows."""
    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
    df: pd.DataFrame = st.session_state.df

    col_stats = get_column_statistics(df)
    example_rows = df.sample(n=min(5, len(df)), random_state=42)

    st.divider()
    with st.expander("See the raw data", expanded=False):
        st.dataframe(df, use_container_width=True)

    with st.expander("Column Summaries", expanded=False):
        st.write(col_stats)

    with st.expander("Example rows", expanded=False):
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

    render_charts(
        suggestions=bar_chart_suggestions,
        df=df,
        generator=bar_chart_data_generator,
        chart_func=px.bar,
        chart_type="bar",
    )
    render_charts(
        suggestions=line_chart_suggestions,
        df=df,
        generator=line_chart_data_generator,
        chart_func=px.line,
        chart_type="line",
    )
    render_charts(
        suggestions=histogram_suggestions,
        df=df,
        generator=histogram_data_generator,
        chart_func=px.histogram,
        chart_type="histogram",
    )
