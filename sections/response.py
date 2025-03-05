import streamlit as st
import pandas as pd
from utils import get_column_statistics, bar_chart_data_generator
from agents.agents import run
import os

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
    example_rows = df.sample(n=min(5, len(df)))

    st.divider()

    with st.expander("See the raw data", expanded=False):
        st.dataframe(df, use_container_width=True)

    with st.expander("Column Summaries", expanded=False):
        st.write(col_stats)

    with st.expander("Example rows", expanded=False):
        # Show as a csv
        st.write(example_rows.to_csv(index=False))

    with st.spinner("Running crew..."):
        data_description_prompt = f"""
        The user has described the data as follows:
        ```
        {st.session_state.data_description}
        ```

        Here are a few randomly chosen examples of the data:
        ```
        {example_rows.to_csv(index=False)}
        ```


        Here are some advanced insights into the data by columns:
        ```json
        {str(col_stats)}
        ```
        """

        with st.expander("Data Description Prompt", expanded=False):
            st.write(data_description_prompt)

        response = run(
            data_description_prompt=data_description_prompt,
            n_visualizations=5,
            llm_id=st.session_state.llm_id,
            step_callback=None,
            task_callback=None,
        )
        with st.expander("Response", expanded=False):
            st.write(response)

    st.divider()

    # chart_configs = [
    #     {
    #         "title": "Top 10 Most Frequent Artists",
    #         "x": "artist_name",
    #         "y": "spotify_track_uri",
    #         "y_agg": "distinct_values",
    #         "y_order": "desc",
    #         "y_label": "Number of Distinct Tracks",
    #         "x_label": "Artist Name",
    #     },
    #     {
    #         "title": "Top 10 Most Played Tracks",
    #         "x": "track_name",
    #         "y": "ms_played",
    #         "y_agg": "sum",
    #         "y_order": "desc",
    #         "y_label": "Total Play Time (ms)",
    #         "x_label": "Track Name",
    #     },
    #     {
    #         "title": "Distribution of Listening Platforms",
    #         "x": "platform",
    #         "y": "spotify_track_uri",
    #         "y_agg": "record_count",
    #         "y_order": "desc",
    #         "y_label": "Number of Plays",
    #         "x_label": "Platform",
    #     },
    #     {
    #         "title": "Reason for Track Start",
    #         "x": "reason_start",
    #         "y": "spotify_track_uri",
    #         "y_agg": "record_count",
    #         "y_order": "desc",
    #         "y_label": "Number of Plays",
    #         "x_label": "Start Reason",
    #     },
    #     {
    #         "title": "Skipped vs Non-Skipped Tracks",
    #         "x": "skipped",
    #         "y": "spotify_track_uri",
    #         "y_agg": "record_count",
    #         "y_order": "desc",
    #         "y_label": "Number of Plays",
    #         "x_label": "Skipped",
    #     },
    # ]

    # rows, cols = 3, 3

    # for row in range(rows):
    #     columns = st.columns(cols)
    #     for col_idx, col in enumerate(columns):
    #         chart_index = row * cols + col_idx
    #         if chart_index < len(chart_configs):
    #             chart_config = chart_configs[chart_index]
    #             with col:
    #                 st.write(f"### {chart_config['title']}")
    #                 st.bar_chart(
    #                     **bar_chart_data_generator(df, chart_config), horizontal=True
    #                 )
