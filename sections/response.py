import streamlit as st
import pandas as pd
from utils import get_column_statistics


def response() -> None:
    """Display raw data, column summaries with statistics, and random full rows.

    Numeric columns: min, max, mean, median, std.
    Datetime columns: min, max.
    Boolean columns: counts of True and False.
    Other columns: number of distinct values and mode.
    """
    df = st.session_state.df

    with st.expander("See the raw data", expanded=True):
        st.dataframe(df, use_container_width=True)

    st.subheader("Column Summaries")
    col_stats = get_column_statistics(df)
    st.write(col_stats)

    st.subheader("Random Full Rows")
    if not df.empty:
        random_rows = df.sample(n=min(5, len(df)))
        st.dataframe(random_rows, use_container_width=True)
