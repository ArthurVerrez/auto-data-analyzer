import streamlit as st
import pandas as pd


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
    for col in df.columns:
        col_data = df[col].dropna()
        if not col_data.empty:
            sample_value = col_data.sample(n=1).iloc[0]
            value_type = type(sample_value).__name__
        else:
            sample_value = None
            value_type = "NoneType"

        distinct = df[col].nunique(dropna=True)
        mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
        num_missing = df[col].isnull().sum()

        # Calculate statistics based on column type.
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                "Name": col,
                "Type": value_type,
                "Example": sample_value,
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Mean": df[col].mean(),
                "Median": df[col].median(),
                "Std": df[col].std(),
                "Distinct Values": distinct,
                "Sum": df[col].sum(),
                "Mode": mode,
                "Missing Values": num_missing,
            }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):

            stats = {
                "Name": col,
                "Type": value_type,
                "Example": sample_value,
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Distinct Values": distinct,
                "Mode": mode,
                "Missing Values": num_missing,
            }
        elif pd.api.types.is_bool_dtype(df[col]):
            true_count = df[col].sum()
            false_count = (~df[col]).sum() if df[col].dtype == bool else None
            stats = {
                "Name": col,
                "Type": value_type,
                "Example": sample_value,
                "True count": true_count,
                "False count": false_count,
                "Missing Values": num_missing,
            }
        # If string, add the average, min, max, median length of the strings
        elif pd.api.types.is_string_dtype(df[col]):
            stats = {
                "Name": col,
                "Type": value_type,
                "Example": sample_value,
                "Min Length": df[col].str.len().min(),
                "Max Length": df[col].str.len().max(),
                "Mean Length": df[col].str.len().mean(),
                "Median Length": df[col].str.len().median(),
                "Distinct Values": distinct,
                "Mode": mode,
                "Missing Values": num_missing,
            }
        else:

            stats = {
                "Name": col,
                "Type": value_type,
                "Example": sample_value,
                "Distinct Values": distinct,
                "Mode": mode,
                "Missing Values": num_missing,
            }
        st.write(stats)

    st.subheader("Random Full Rows")
    if not df.empty:
        random_rows = df.sample(n=min(5, len(df)))
        st.dataframe(random_rows, use_container_width=True)
