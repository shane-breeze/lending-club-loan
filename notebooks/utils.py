from itertools import combinations

import numpy as np
import pandas as pd

def highlight_nan(series, perc=0.01):
    """
    highlight rows in red with count<perc*nrows, yellow if count!=nrows and
    otherwise no colouring
    """
    return [
        'background-color: #fb9a99' if series["nan"] > perc*series["rows"] else
        'background-color: #fdbf6f' if series["count"] != series["rows"] else
        ''
    ]*series.shape[0]

def highlight_dtype(series, dtype='object'):
    """highlight rows in blue if dtype==dtype"""
    return [
        'color: #1f78b4' if series["dtype"] == dtype else ''
    ]*series.shape[0]

def duplicate_columns(dframe):
    """Return tuple of column pairs which are identical"""
    return [
        (i, j) for i, j in combinations(dframe, 2)
        if dframe[i].equals(dframe[j])
    ]

def convert_str_to_month_delta(dframe, columns):
    """
    Convert columns to datetime64 then subtract unix time and convert
    to month delta (year - 1970)*12 + (month - 1)
    """
    dates = []
    for col in columns:
        date = dframe[col].astype("datetime64")
        dates.append((date.dt.year - 1970)*12 + (date.dt.month - 1))
    return pd.concat(dates, axis=1)

def generate_stats(df, nunique=False):
    df_vals = df.loc[:,(df.dtypes==np.float) | (df.dtypes==np.int)]

    # caution: nunique is slow for large dataframes
    data = {
        "dtype": df.dtypes,
        "rows": [df.shape[0]]*df.shape[1],
        "count": df.count(),
        "nan": df.isna().sum(),
        "inf": np.isinf(df_vals).sum(),
        "mean": df_vals.mean(),
        "std": df_vals.std(),
        "min": df_vals.min(),
        "25%": df_vals.quantile(0.25),
        "50%": df_vals.quantile(0.5),
        "75%": df_vals.quantile(0.75),
        "max": df_vals.max(),
        "1st": df.iloc[0,:],
        "2nd": df.iloc[1,:],
        "3rd": df.iloc[2,:],
    }
    if nunique:
        data["nuniq"] = df.nunique()
    stats = pd.DataFrame(data).fillna('').astype({"dtype": "str"})
    stats.index = pd.Categorical(stats.index, df.columns)
    return stats.sort_index()