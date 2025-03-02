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
