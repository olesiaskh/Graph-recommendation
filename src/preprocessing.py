import pandas as pd


def add_prefixes(df):
    """Add u to user_id and v to venue_id to differentiate betweeen
    them in the graph"""
    if "user_id" in df.columns:
        df["user_id"] = "u" + df["user_id"].astype(str)
    if "venue_id" in df.columns:
        df["venue_id"] = "v" + df["venue_id"].astype(str)
    return df


def all_unique_values(dfs, column):
    """Get a set of all unique values from multiple dataframes"""
    values = set()
    for df in dfs:
        values.update(set(df[column]))

    return values
