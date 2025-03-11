import streamlit as st
import plotly.express as px
from utils import get_df_from_file, bar_chart_data_generator, line_chart_data_generator

df = get_df_from_file("data/spotify_history.csv")
bar_chart_config = {
    "title": "Top 10 Most Played Tracks",
    "x": "track_name",
    "y": "ms_played",
    "y_agg": "sum",
    "y_order": "desc",
    "x_label": "Track Name",
    "y_label": "Total Play Time (ms)",
}

st.plotly_chart(px.bar(**bar_chart_data_generator(df, bar_chart_config)))


line_chart_config = {
    "title": "Total Playtime by Track",
    "x": "ts",
    "y": "ms_played",
    "x_time_trunc": "month",
    "y_agg": "sum",
    "y_order": "desc",
    "x_label": "Date",
    "y_label": "Total Playtime (ms)",
}

st.plotly_chart(px.line(**line_chart_data_generator(df, line_chart_config)))


# st.plotly_chart(px.histogram(**histogram_data_generator(df, histogram_config)))
