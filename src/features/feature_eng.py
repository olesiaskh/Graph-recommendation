import numpy as np
from networkx import nx
from review_features import (
    avg_visits,
    encode_tags,
    common_tags,
    get_all_venue_sentiment,
    get_all_visit_counts,
    get_top_tags,
)
from network_similarity import jaccard, adamic_adar, common_neighbors


def feature_extractor(graph, samples, df_sentiment, df_chekins, df_tags, labels=None):
    """
    Create feature vectors to be used for classification
    """
    # prepare sentiment values dict for encoding
    venue_sent_d, default_sent = get_all_venue_sentiment(df_sentiment)

    # prepare visit counts dict for encoding
    all_visits = get_all_visit_counts(df_chekins)

    # prepare to tags dict for processing
    top_tags = get_top_tags(df_tags)

    # average sentiment towards the venue
    sent = []

    # average  number of visit to the venue
    n_visits = []

    # tags of the venue
    # tag = []

    # tag overlap between the venue and the user's past
    common_tag = []

    # degree of the user
    deg_u = []

    # degree of the venue
    deg_v = []

    # adamic adar on sets of users
    adam_u = []

    # adamic adar on sets of venues
    adam_v = []

    # jaccard similarity on sets of users
    # jacc_u = []

    # jaccard similarity on sets of venues
    jacc_v = []

    # common neighbors on sets of users
    # cn_u = []

    # common neighbors on sets of venues
    # cn_v = []

    counter = 0

    # FEATURES PREPROCESSING
    for i, sample in enumerate(samples):
        user = sample[0]  # id user
        venue = sample[1]  # id venue

        # REMOVE THE EDGE, otherwise we will bias the computations of distance metrics (we won't have connected edges in the test set)
        # do this only if labels are provided (function applied to train set)
        if labels is not None and labels[i] == 1:
            graph.remove_edge(user, venue)

        # average sentiment towards the venue
        sent.append(venue_sent_d.get(venue, default_sent))

        # average  number of visit to the venue
        n_visits.append(avg_visits(venue, all_visits))

        # tags of the venue
        # tag.append(encode_tags(venue, df_tags, top_tags).tolist())

        # tag overlap between the venue and the user's past
        common_tag.append(common_tags(graph, user, venue, df_tags))

        # degree of the user
        deg_u.append(graph.degree[user])

        # degree of the venue
        deg_v.append(graph.degree[venue])

        # jaccard similarity on sets of users
        # jacc_u.append(jaccard(graph, user, venue))

        # jaccard similarity on sets of venues
        jacc_v.append(jaccard(graph, venue, user))

        # adamic adar on sets of users
        adam_u.append(adamic_adar(graph, user, venue))

        # adamic adar on sets of venues
        adam_v.append(adamic_adar(graph, venue, user))

        # common neighbors on sets of users
        # cn_u.append(common_neighbors(graph, user, venue))

        # common neighbors on sets of venues
        # cn_v.append(common_neighbors(graph, venue, user))

        # We now re-add the edge to our graph
        if labels is not None and labels[i] == 1:
            graph.add_edge(user, venue)

        if counter % 500:
            print(counter, "examples processed")
        counter += 1

    # Convert list of features into array
    feature_vector = [
        n_visits,
        sent,
        deg_v,
        deg_u,
        adam_u,
        adam_v,
        jacc_v,
        common_tag,
    ]  # if include tag vectors, put it as the last feature in the vector
    feature_vector = np.array(feature_vector).T
    return feature_vector


def reformat_tags(feature_vector):
    """If tag vector is included in the feature vector, flatten it"""
    v_len = len(feature_vector[0]) - 1

    feature_vector = feature_vector.tolist()
    for i, vector in enumerate(feature_vector):
        feature_vector[i] = vector[:v_len] + vector[-1]
    feature_vector = np.array(feature_vector)

    return feature_vector
