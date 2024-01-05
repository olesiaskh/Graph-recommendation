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


def str_to_list(df, column):
    """Convert values in a coluumn from string to list"""
    df[column] = df[column].apply(lambda s: s.split(",") if isinstance(s, str) else s)
    return df


def clean_tags(df_tags):
    """Clean tags dataframe - drop NAs and convert values to list"""
    df_tags = df_tags.dropna().reset_index(drop=True)
    df_tags = str_to_list(df_tags, "tag_list")
    return df_tags


def clean_sentiment(df_sentiment):
    """Clean sentiment dataframe - drop unnecessary columns"""
    return df_sentiment.drop(columns=["Unnamed: 0", "user_id", "tip"])
