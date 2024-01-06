import networkx as nx
import pandas as pd

from preprocessing import add_prefixes, all_unique_values
from features.feature_eng import feature_extractor
from features.network_similarity import get_gcc
from features.sentiment import calculate_review_sentiment, calculate_affinity
from training import (
    generate_samples,
    compare_models_supervised,
    compare_models_unsupervised,
)


default_path = "data/"


if __name__ == "__main__":
    # Import original data
    df_chekins = pd.read_csv(
        default_path + "dataset_ubicomp2013_checkins.txt",
        sep="\t",
        names=["user_id", "venue_id"],
    )
    df_tags = pd.read_csv(
        default_path + "dataset_ubicomp2013_tags.txt",
        sep="\t",
        names=["venue_id", "tag_list"],
        encoding="unicode_escape",
    )
    df_tips = pd.read_csv(
        default_path + "dataset_ubicomp2013_tips.txt",
        sep="\t",
        names=["user_id", "venue_id", "tip_text"],
        encoding="unicode_escape",
    )

    # to distinguish between users and venues in the future, add prefixes to ids (u and v, respectively)
    df_chekins = add_prefixes(df_chekins)
    df_tips = add_prefixes(df_tips)
    df_tags = add_prefixes(df_tags)

    # calculate sentiment of each review
    df_sentiment = calculate_review_sentiment(df_tips)

    # transform to affinity scores between user-venue pairs (to create only edges of a positive experience)
    df_scores = calculate_affinity(df_sentiment, df_chekins)

    # get sets of venues and users from all datasets
    users = all_unique_values([df_chekins, df_tips], "user_id")
    venues = all_unique_values([df_chekins, df_tips, df_tags], "venue_id")

    # Construct the graph

    B = nx.Graph()
    B.add_nodes_from(users, bipartite=0)  # top nodes - users
    B.add_nodes_from(venues, bipartite=1)  # bottom nodes - venues

    # Add edges that indicate positive experience (affinity score at least 3)
    for index, row in df_scores.iterrows():
        if row["weights"] >= 3:
            B.add_edge(row["user_id"], row["venue_id"])

    # Get GCC
    B_gcc = get_gcc(graph=B)

    # Generate train and test sets
    (
        residual_g,
        train_samples,
        train_labels,
        test_samples,
        test_labels,
    ) = generate_samples(graph=B_gcc, train_set_ratio=0.7)

    # Create feature vectors for train and test set
    train_features = feature_extractor(
        residual_g,
        train_samples,
        df_sentiment,
        df_chekins,
        df_tags,
        labels=train_labels,
    )
    test_features = feature_extractor(
        residual_g,
        test_samples,
        df_sentiment,
        df_chekins,
        df_tags,
    )

    # Train the model and compare results
    # compare_models_unsupervised(residual_g, test_samples, test_labels)
    compare_models_supervised(train_features, test_features, train_labels, test_labels)
