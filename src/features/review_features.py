from collections import defaultdict
import numpy as np
import nltk
from preprocessing import clean_tags, add_prefixes, clean_sentiment

nltk.download("wordnet")


def get_tag_frequency(df_tags):
    """Create a dictionary of tag frequency (lemmatized)
    Outputs dictionary of tags, where (k=tag, v=number of restaurants with this tag)
    """
    lemma = nltk.wordnet.WordNetLemmatizer()
    d_tags = defaultdict(int)

    df_tags = clean_tags(df_tags)

    for i, row in df_tags.iterrows():
        tags_list = row["tag_list"]
        for tag in tags_list:
            tag = tag.lstrip()
            words = tag.split(" ")
            tag_clean = ""
            word_count = 0
            for word in words:
                word_clean = lemma.lemmatize(word)
                if word_count >= 1:
                    tag_clean += " " + word_clean
                else:
                    tag_clean += word_clean
                word_count += 1
                d_tags[tag_clean] += 1

    return d_tags


def get_top_tags(d_tags, min_count=150):
    """Get a dictionary of tags that appear for at least min_count reastaurants"""
    top_tags = {}
    for tag, freq in d_tags.items():
        if freq >= min_count:
            top_tags[tag] = freq
    return top_tags


def tag_vector_ids(top_tags):
    """Create a mapping between top tags and their location in a vector"""
    tag_codes = {}
    i = 0
    for tag in top_tags:
        tag_codes[tag] = i
        i += 1
    return tag_codes


def encode_tags(venue, df_tags, top_tags):
    """
    Create a one-hot encoded vector of venue tags (only top tags are encoded)
    """
    tagged_venues = set(clean_tags(df_tags).venue_id)
    tag_codes = tag_vector_ids(top_tags)
    tags_vector = np.zeros(len(top_tags))

    if venue in tagged_venues:
        tags = df_tags.loc[df_tags["venue_id"] == venue, "tag_list"].iloc[0]
        for tag in tags:
            if tag in top_tags:
                tags_vector[tag_codes.get(tag)] = 1
    return tags_vector


def common_tags(graph, user, venue, df_tags):
    """
    Calculate the proportion of common tags between venue and user's already visited venues
    """
    tagged_venues = set(clean_tags(df_tags).venue_id)
    # get all user's visited tags
    visited_ven = graph.neighbors(user)
    user_tag_list = set()
    for ven in visited_ven:
        if ven in tagged_venues:
            tags = df_tags.loc[df_tags["venue_id"] == ven, "tag_list"].iloc[0]
            user_tag_list.update(set(tags))

    # get all venue tags
    if venue in tagged_venues:
        venue_tag_list = set(
            df_tags.loc[df_tags["venue_id"] == venue, "tag_list"].iloc[0]
        )
        return len(user_tag_list.intersection(venue_tag_list)) / len(venue_tag_list)
    else:
        return 0  # if venue does not have any tags


def get_all_venue_sentiment(df_sentiment):
    """Get a dictionary of average sentiment score for each venue id,
    and average sentiment for default
    The dataframe has pre-calculated average scores for each user"""
    venue_sent = clean_sentiment(df_sentiment)
    venue_sent = add_prefixes(venue_sent)
    venue_sent_d = venue_sent.groupby(["venue_id"]).mean().to_dict()
    venue_sent_d = venue_sent_d.get("sentiment")
    default_sent = np.mean(list(venue_sent_d.values()))
    return venue_sent_d, default_sent


def get_all_visit_counts(df_chekins):
    """Create a dictionary with a list of total visit count per user for each venue"""
    # First, create a dictionary of number of user-venues visits
    # uv_visists(k=(user, venue), v=number of visits)

    uv_visits = defaultdict(int)
    for i, row in df_chekins.iterrows():
        visit = (row["user_id"], row["venue_id"])
        uv_visits[visit] += 1

    # Second, create a dictionary of number of visists of alll user to the venue
    # v_visits(k=venue, v=[number of visits for all users who went there before])
    v_visits = defaultdict(list)
    for k, v in uv_visits.items():
        v_visits[k[1]].append(v)

    return v_visits


def avg_visits(venue, all_visits):
    """
    Calculate how many times people visit the venue, on average
    """
    visit_list = all_visits.get(venue)
    if visit_list:
        return np.mean(visit_list)
    else:
        return 0
