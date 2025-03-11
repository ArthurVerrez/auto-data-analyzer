import pandas as pd
import numpy as np

from constants import (
    MAX_POINTS_BAR_CHART,
    MAX_POINTS_LINE_CHART,
    MAX_STRING_LABEL_LENGTH,
    MAX_BINS_HISTOGRAM,
)


def default_np_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def timestamp_converter(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def get_df_from_file(f) -> pd.DataFrame:
    """Read CSV file and auto-detect better types for object columns.

    Tries to convert object columns to datetime, numeric, or boolean.
    If conversion fails, leaves column as object.

    Args:
        f: File-like object or path to a CSV file.

    Returns:
        DataFrame with improved data types.
    """
    df = pd.read_csv(f)

    for col in df.select_dtypes(include=["object"]).columns:
        # Try converting to datetime.
        try:
            converted = pd.to_datetime(
                df[col], errors="raise", infer_datetime_format=True
            )
            df[col] = converted
            continue
        except Exception:
            pass

        # Try converting to numeric.
        try:
            converted = pd.to_numeric(df[col], errors="raise")
            df[col] = converted
            continue
        except Exception:
            pass

        # Try converting to boolean if the column has only boolean-like values.
        unique_vals = df[col].dropna().unique()
        bool_map = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
        lower_unique_vals = {str(val).strip().lower() for val in unique_vals}
        if lower_unique_vals.issubset(set(bool_map.keys())):
            df[col] = df[col].apply(lambda x: bool_map.get(str(x).strip().lower(), x))

    return df


def get_column_statistics(df):
    col_stats = []
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
        stats = {k: default_np_converter(v) for k, v in stats.items()}
        col_stats.append(stats)
    return col_stats


def bar_chart_data_generator(df: pd.DataFrame, chart_config: dict) -> dict:
    """Generate parameters for a horizontal bar chart with Plotly.

    Aggregates data based on the provided configuration and returns a dictionary
    with parameters to be unpacked into px.bar().

    Args:
        df (pd.DataFrame): Input DataFrame.
        chart_config (dict): Chart configuration containing:
            - x (str): Column name for categorical data.
            - y (str): Column name for numerical data.
            - y_agg (str): Aggregation function ("distinct_values", "sum", "record_count", "median").
            - y_order (str, optional): Order for the y axis ("asc", "desc", "rand"). Defaults to "desc".
            - x_order (str, optional): Secondary order for the x axis ("asc", "desc", "rand").
            - title (str, optional): Chart title.
            - x_label (str, optional): Label for the categorical axis.
            - y_label (str, optional): Label for the numerical axis.

    Returns:
        dict: Dictionary with keys:
            - data_frame (pd.DataFrame): Aggregated DataFrame.
            - x (str): Column for x-axis (numeric value).
            - y (str): Column for y-axis (categorical value).
            - title (str): Chart title.
            - labels (dict): Mapping of axis names to labels.
            - orientation (str): Chart orientation ("h").
    """
    x_col = chart_config["x"]
    y_col = chart_config["y"]
    agg_func = chart_config["y_agg"]
    y_order = chart_config.get("y_order", "desc")
    x_order = chart_config.get("x_order", None)

    if agg_func == "distinct_values":
        agg_series = df.groupby(x_col)[y_col].nunique()
    elif agg_func == "sum":
        agg_series = df.groupby(x_col)[y_col].sum()
    elif agg_func == "record_count":
        agg_series = df.groupby(x_col).size()
    elif agg_func == "median":
        agg_series = df.groupby(x_col)[y_col].median()
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")

    agg_df = agg_series.reset_index(name=y_col)

    if y_order == "asc":
        agg_df = agg_df.sort_values(by=y_col, ascending=True)
    elif y_order == "desc":
        agg_df = agg_df.sort_values(by=y_col, ascending=False)
    elif y_order == "rand":
        agg_df = agg_df.sample(frac=1)
    else:
        raise ValueError(f"Unsupported y_order value: {y_order}")

    if x_order and x_order != "rand":
        if x_order == "asc":
            agg_df = agg_df.sort_values(
                by=[y_col, x_col], ascending=[y_order == "asc", True]
            )
        elif x_order == "desc":
            agg_df = agg_df.sort_values(
                by=[y_col, x_col], ascending=[y_order == "asc", False]
            )
        else:
            raise ValueError(f"Unsupported x_order value: {x_order}")

    agg_df = agg_df.head(MAX_POINTS_BAR_CHART)

    # Truncate long string labels to MAX_STRING_LABEL_LENGTH
    if pd.api.types.is_string_dtype(agg_df[x_col]):
        agg_df[x_col] = agg_df[x_col].apply(
            lambda x: (
                x[:MAX_STRING_LABEL_LENGTH] + "..."
                if len(x) > MAX_STRING_LABEL_LENGTH
                else x
            )
        )

    agg_df[x_col] = pd.Categorical(
        agg_df[x_col], categories=agg_df[x_col].tolist(), ordered=True
    )

    return {
        "data_frame": agg_df,
        "x": y_col,  # Numeric value for horizontal bar length
        "y": x_col,  # Categorical value for bar labels
        "title": chart_config.get("title", ""),
        "labels": {
            y_col: chart_config.get("y_label", y_col),
            x_col: chart_config.get("x_label", x_col),
        },
        "orientation": "h",
    }


def line_chart_data_generator(df: pd.DataFrame, chart_config: dict) -> dict:
    """Generate parameters for a Plotly line chart from a dataframe based on chart configuration.

    Aggregates data based on the provided configuration and returns a dictionary
    with keys for use with px.line().

    Args:
        df (pd.DataFrame): Input DataFrame.
        chart_config (dict): Chart configuration containing:
            - "title": chart title.
            - "x": column name for x-axis.
            - "y": column name for y-axis.
            - "y_agg": one of "distinct_values", "sum", "record_count", "median".
            - "x_order": optional, one of "asc", "desc", "rand". Defaults to "asc".
            - "y_order": optional, one of "asc", "desc", "rand". If provided, ordering is applied on the y value.
            - "x_label": optional, label for x-axis.
            - "y_label": optional, label for y-axis.
            - "x_time_trunc": optional, one of "s", "min", "hour", "day", "week", "month", "year".
              Applied only if the x column is a time or datelike type.

    Returns:
        dict: Dictionary with keys:
            - data_frame (pd.DataFrame): Aggregated DataFrame.
            - x (str): Column for x-axis.
            - y (str): Column for y-axis.
            - title (str): Chart title.
            - labels (dict): Mapping of axis names to labels.
    """
    x_col = chart_config["x"]
    y_col = chart_config["y"]
    agg_func = chart_config["y_agg"]
    x_order = chart_config.get("x_order", "asc")
    y_order = chart_config.get("y_order", None)

    if "x_time_trunc" in chart_config and pd.api.types.is_datetime64_any_dtype(
        df[x_col]
    ):
        mapping = {
            "s": "S",
            "min": "T",
            "hour": "H",
            "day": "D",
            "week": "W",
            "month": "M",
            "year": "Y",
        }
        trunc_value = chart_config["x_time_trunc"].lower()
        if trunc_value in mapping:
            freq = mapping[trunc_value]
            if trunc_value == "month":
                group_key = df[x_col].dt.to_period("M").dt.to_timestamp()
            elif trunc_value == "year":
                group_key = df[x_col].dt.to_period("Y").dt.to_timestamp()
            elif trunc_value == "week":
                group_key = df[x_col].dt.to_period("W").dt.to_timestamp()
            else:
                group_key = df[x_col].dt.floor(freq)
        else:
            group_key = df[x_col]
    else:
        group_key = df[x_col]

    if agg_func == "distinct_values":
        agg_series = df.groupby(group_key)[y_col].nunique()
    elif agg_func == "sum":
        agg_series = df.groupby(group_key)[y_col].sum()
    elif agg_func == "record_count":
        agg_series = df.groupby(group_key).size()
    elif agg_func == "median":
        agg_series = df.groupby(group_key)[y_col].median()
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")

    agg_df = agg_series.reset_index(name=y_col)

    # Apply ordering: if y_order is specified, sort by y value first, then by x if provided.
    if y_order is not None:
        if y_order == "asc":
            agg_df = agg_df.sort_values(by=y_col, ascending=True)
        elif y_order == "desc":
            agg_df = agg_df.sort_values(by=y_col, ascending=False)
        elif y_order == "rand":
            agg_df = agg_df.sample(frac=1)
        else:
            raise ValueError(f"Unsupported y_order value: {y_order}")

        if x_order and x_order != "rand":
            if x_order == "asc":
                agg_df = agg_df.sort_values(
                    by=[y_col, x_col], ascending=[y_order == "asc", True]
                )
            elif x_order == "desc":
                agg_df = agg_df.sort_values(
                    by=[y_col, x_col], ascending=[y_order == "asc", False]
                )
            else:
                raise ValueError(f"Unsupported x_order value: {x_order}")
    else:
        # Fallback to x_order if no y_order is specified.
        if x_order == "asc":
            agg_df = agg_df.sort_values(by=x_col, ascending=True)
        elif x_order == "desc":
            agg_df = agg_df.sort_values(by=x_col, ascending=False)
        elif x_order == "rand":
            agg_df = agg_df.sample(frac=1)
        else:
            raise ValueError(f"Unsupported x_order value: {x_order}")

    agg_df = agg_df.head(MAX_POINTS_LINE_CHART)

    if pd.api.types.is_string_dtype(agg_df[x_col]):
        agg_df[x_col] = agg_df[x_col].apply(
            lambda x: (
                x[:MAX_STRING_LABEL_LENGTH] + "..."
                if len(x) > MAX_STRING_LABEL_LENGTH
                else x
            )
        )

    return {
        "data_frame": agg_df,
        "x": x_col,
        "y": y_col,
        "title": chart_config.get("title", ""),
        "labels": {
            x_col: chart_config.get("x_label", x_col),
            y_col: chart_config.get("y_label", y_col),
        },
    }


def histogram_data_generator(df: pd.DataFrame, histogram_config: dict) -> dict:
    """
    Generate configuration for a histogram visualization using Plotly Express.

    Args:
        df (pd.DataFrame): DataFrame to visualize.
        histogram_config (dict): Dictionary containing histogram configuration.
            Expected keys:
                - "x": Column name for the x-axis.
                - "bins": Number of bins for the histogram.
                - "title": (Optional) Chart title.
                - "x_label": (Optional) Human-readable label for the x-axis.
                - "y_label": (Optional) Human-readable label for the y-axis.

    Returns:
        dict: Dictionary of configuration parameters for px.histogram.
    """
    config = {
        "data_frame": df,
        "x": histogram_config.get("x"),
        "nbins": histogram_config.get("bins"),
        "title": histogram_config.get("title"),
    }
    labels = {}
    if "x_label" in histogram_config:
        labels[histogram_config["x"]] = histogram_config["x_label"]
    if "y_label" in histogram_config:
        labels["count"] = histogram_config["y_label"]
    if labels:
        config["labels"] = labels
    return config


def display_grid_streamlit(
    chart_suggestions, df, chart_type, chart_data_generator, px_chart_function
):
    """Display a grid of charts in Streamlit.

    Args:
        chart_suggestions: List of chart configurations
        df: DataFrame containing the data
        chart_type: Type of chart ('bar' or 'line')
        chart_data_generator: Function to generate chart data
        px_chart_function: Plotly Express chart function (px.bar or px.line)
    """
    import streamlit as st

    rows, cols = (len(chart_suggestions) + 2) // 3, 3

    for row in range(rows):
        columns = st.columns(cols)
        for col_idx, col in enumerate(columns):
            chart_index = row * cols + col_idx
            if chart_index < len(chart_suggestions):
                chart_config = chart_suggestions[chart_index]
                with col:
                    try:
                        st.write(f"### {chart_config['title']}")
                        st.plotly_chart(
                            px_chart_function(**chart_data_generator(df, chart_config))
                        )
                    except Exception as e:
                        st.write(
                            f"Error: {e} occurred while rendering the chart. Please try another chart suggestion."
                        )
