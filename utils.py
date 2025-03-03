import pandas as pd


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
        col_stats.append(stats)
    return col_stats
