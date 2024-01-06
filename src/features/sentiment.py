import pandas as pd
from transformers import pipeline


def get_weighted_sentiment(sent_prob):
    """Calculate weighted average of sentiment"""
    return sum(int(line["label"].split()[0]) * line["score"] for line in sent_prob)


def calculate_review_sentiment(df_tips):
    """Get weighted sentiment score for each review"""
    classifier = pipeline(
        "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    classifier.return_all_scores = True  # need to get all probs for weighted average
    df_sentiment = df_tips.copy()

    for i, row in df_sentiment.iterrows():
        result = classifier(row["tip_text"])[0]
        df_sentiment.loc[i, "sentiment"] = get_weighted_sentiment(result)

    return df_sentiment


def calculate_affinity(df_sentiment, df_chekins):
    """Calculate affinity score for each venue based on sentiment of review and number of visits
    using the following logic:

    If
    Number of visits (check-ins) <= Number_sentiments : implies the weight is the sum of the sentiments

    Else if:
    the excess of check-ins are evaluated with the user average (if available)

    Else if:
    if user average not available, the excess is evaluated using the venue average

    Else:
    excess is evaluated as 3
    """
    # calculate average sentiment for each venue and user
    avg_sent_venue = (
        df_sentiment[["venue_id", "sentiment"]].groupby(["venue_id"]).mean()
    )
    avg_sent_user = df_sentiment[["user_id", "sentiment"]].groupby(["user_id"]).mean()

    # calculate total number of visits for each user to each venue
    df_chekins["visits"] = 1
    number_visits = df_chekins.groupby(["user_id", "venue_id"]).count()

    # get number of reviews per user per venue
    number_sentiments_user = (
        df_sentiment[["user_id", "venue_id", "sentiment"]]
        .groupby(["user_id", "venue_id"])
        .count()
    )

    # sum up sentiment per user per venue
    total_sentiment = (
        df_sentiment[["user_id", "venue_id", "sentiment"]]
        .groupby(["user_id", "venue_id"])
        .sum()
    )

    # get all user and venues for which info is available
    affinity_df = pd.concat(
        (df_sentiment[["user_id", "venue_id"]], df_chekins[["user_id", "venue_id"]])
    ).drop_duplicates()

    # use the information to calculate affinity score
    for i, row in affinity_df.iterrows():
        user_id = row[0]
        venue_id = row[1]
        try:
            no_visits = number_visits.loc[(user_id, venue_id), "visits"]
        except KeyError:
            no_visits = 0
        try:
            no_sentiments = number_sentiments_user.loc[(user_id, venue_id), "sentiment"]
        except KeyError:
            no_sentiments = 0
        try:
            average_user = avg_sent_user.loc[user_id, "sentiment"]
        except KeyError:
            try:
                average_user = avg_sent_venue.loc[venue_id, "sentiment"]
            except KeyError:
                average_user = 3
        if no_visits <= no_sentiments:
            weights = total_sentiment.loc[(user_id, venue_id), "sentiment"]
        else:
            try:
                weights = (
                    total_sentiment.loc[(user_id, venue_id), "sentiment"]
                    + (no_visits - no_sentiments) * average_user
                )
            except KeyError:
                weights = (no_visits - no_sentiments) * average_user

        affinity_df.loc[i, "weights"] = weights

    return affinity_df
