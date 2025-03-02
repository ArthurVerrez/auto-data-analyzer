import pandas as pd


def get_df_from_file(f):
    df = pd.read_csv(f)
    return df
