import pandas as pd
import numpy as np


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
        # Apply the default_np_converter to each value in the dictionary
        stats = {k: default_np_converter(v) for k, v in stats.items()}
        col_stats.append(stats)
    return col_stats


def bar_chart_data_generator(df: pd.DataFrame, chart_config: dict) -> dict:
    """Generate parameters for st.bar_chart from a dataframe based on chart configuration.

    Args:
        df (pd.DataFrame): Input DataFrame.
        chart_config (dict): Chart configuration containing:
            - "title": chart title.
            - "x": column name for x-axis.
            - "y": column name for y-axis.
            - "y_agg": one of "distinct_values", "sum", "record_count", "median".
            - "y_order": optional, one of "asc", "desc", "rand". Defaults to "desc".
            - "x_order": optional, one of "asc", "desc", "rand".
            - "y_label": optional, label for y-axis.
            - "x_label": optional, label for x-axis.

    Returns:
        dict: Dictionary with key "data" as a DataFrame suitable for st.bar_chart,
            and optional "x_label" and "y_label" keys.
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

    # Order by aggregated y value first
    if y_order == "asc":
        agg_df = agg_df.sort_values(by=y_col, ascending=True)
    elif y_order == "desc":
        agg_df = agg_df.sort_values(by=y_col, ascending=False)
    elif y_order == "rand":
        agg_df = agg_df.sample(frac=1)
    else:
        raise ValueError(f"Unsupported y_order value: {y_order}")

    # Limit to top 10 entries after y_order sorting
    agg_df = agg_df.head(10)

    # If x_order is provided, re-order the x labels accordingly
    if x_order:
        if x_order == "asc":
            agg_df = agg_df.sort_values(by=x_col, ascending=True)
        elif x_order == "desc":
            agg_df = agg_df.sort_values(by=x_col, ascending=False)
        elif x_order == "rand":
            agg_df = agg_df.sample(frac=1)
        else:
            raise ValueError(f"Unsupported x_order value: {x_order}")

    # Set the index as a categorical index to preserve ordering in Streamlit's bar_chart
    agg_df.set_index(x_col, inplace=True)
    agg_df.index = pd.CategoricalIndex(
        agg_df.index, categories=agg_df.index.tolist(), ordered=True
    )

    output = {"data": agg_df}
    if "x_label" in chart_config:
        output["x_label"] = chart_config["x_label"]
    else:
        output["x_label"] = x_col
    if "y_label" in chart_config:
        output["y_label"] = chart_config["y_label"]
    else:
        output["y_label"] = y_col
    return output
