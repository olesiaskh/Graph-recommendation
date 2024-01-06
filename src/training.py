import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.inspection import permutation_importance
from features.network_similarity import (
    common_neighbors_pred,
    jaccard_pred,
    adamic_adar_pred,
)


def generate_samples(graph, train_set_ratio):
    """
    Create training and test sets (samples and labels), making sure graph stays connected
    """

    # --- Step 0: The graph must be connected ---
    if nx.is_connected(graph) is not True:
        raise ValueError("The graph contains more than one connected component!")

    # --- Step 1: Generate positive edge samples for testing set ---
    residual_g = graph.copy()
    test_pos_samples = []

    # Store the shuffled list of current edges of the graph
    edges = list(residual_g.edges())
    np.random.shuffle(edges)

    # Define number of positive test samples desired
    test_set_size = int((1.0 - train_set_ratio) * graph.number_of_edges())
    num_of_pos_test_samples = 0

    # Remove random edges from the graph, leaving it connected
    for edge in edges:
        # Remove the edge
        residual_g.remove_edge(edge[0], edge[1])

        # Add the removed edge to the positive sample list if the network is still connected
        # AND if the edge is in user-venue format
        if nx.is_connected(residual_g) and edge[0][0] == "u" and edge[1][0] == "v":
            num_of_pos_test_samples += 1
            test_pos_samples.append(edge)
        # Otherwise, re-add the edge to the network
        else:
            residual_g.add_edge(edge[0], edge[1])

        # If we have collected enough number of edges for testing set, we can terminate the loop
        if num_of_pos_test_samples == test_set_size:
            break

    # Check if we have the desired number of positive samples for testing set
    if num_of_pos_test_samples != test_set_size:
        raise ValueError("Enough positive edge samples could not be found!")

    # --- Step 2: Generate positive edge samples for training set ---
    # The remaining edges are simply considered for positive samples of the training set
    train_pos_samples = []
    for edge in residual_g.edges():
        # Again, make sure that the edges added are only in user-venue format
        if edge[0][0] == "u" and edge[1][0] == "v":
            train_pos_samples.append(edge)

    # --- Step 3: Generate the negative samples for testing and training sets ---
    non_edges = list(nx.non_edges(graph))
    np.random.shuffle(non_edges)
    num_of_neg_samples = 0
    # To make balanced datasets, take the same number of negative edges as positive edges
    size = len(train_pos_samples) + len(test_pos_samples)
    neg_samples = []

    for non_edge in non_edges:
        # If non-edges are in user-venue format, add them to the list of negative samples
        if non_edge[0][0] == "u" and non_edge[1][0] == "v":
            neg_samples.append(non_edge)
            num_of_neg_samples += 1

        # If we have collected enough number of negative samples, we can terminate the loop
        if num_of_neg_samples == size:
            break

    # Split the total set of negative samples into train and test
    train_neg_samples = neg_samples[: len(train_pos_samples)]
    test_neg_samples = neg_samples[len(train_pos_samples) :]

    # --- Step 4: Combine sample lists and create corresponding labels ---
    # For training set
    train_samples = train_pos_samples + train_neg_samples
    train_labels = [1 for _ in train_pos_samples] + [0 for _ in train_neg_samples]
    # For testing set
    test_samples = test_pos_samples + test_neg_samples
    test_labels = [1 for _ in test_pos_samples] + [0 for _ in test_neg_samples]

    return residual_g, train_samples, train_labels, test_samples, test_labels


def get_roc(test_labels, test_preds, show_curve=False):
    """
    Get roc_auc score for predictions and (optionally) plot roc curve
    """
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    roc_auc = auc(fpr, tpr)
    if show_curve:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="darkred", label="ROC curve (area = %0.3f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="lightgray", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic Curve")
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc


def compare_models_unsupervised(residual_g, test_samples, test_labels):
    """Use multiple unsupervised models to predict links and compare results"""
    # Jaccard on users
    jaccard_pred_u = jaccard_pred(residual_g, test_samples, users=True)
    print(f"Jaccard User: {get_roc(test_labels, jaccard_pred_u)}")

    # Jaccard on venues
    jaccard_pred_v = jaccard_pred(residual_g, test_samples, users=False)
    print(f"Jaccard Venue: {get_roc(test_labels, jaccard_pred_v)}")

    # Adamic Adar on users
    ad_ar_u = adamic_adar_pred(residual_g, test_samples, users=True)
    print(f"AdamicAdar User: {get_roc(test_labels, ad_ar_u)}")

    # Adamic Adar on venues
    ad_ar_v = adamic_adar_pred(residual_g, test_samples, users=False)
    print(f"AdamicAdar Venue: {get_roc(test_labels, ad_ar_v)}")

    # Common Neighbors on users
    cn_u = common_neighbors_pred(residual_g, test_samples, users=True)
    print(f"Common Neighbor User: {get_roc(test_labels, cn_u)}")

    # Common Neighbors on venues
    cn_v = common_neighbors_pred(residual_g, test_samples, users=False)
    print(f"Common Neighbor Venue: {get_roc(test_labels, cn_v)}")


def compare_models_supervised(train_features, test_features, train_labels, test_labels):
    """Train multiple supervised models and compare results"""
    # MODELS
    SVM = SVC()
    Logistic = LogisticRegression(max_iter=2000)
    RandomForest = RandomForestClassifier()
    AdaBoost = AdaBoostClassifier()
    xgb = XGBClassifier()
    xtra = ExtraTreesClassifier()

    # SVM
    train_features_prep = preprocessing.scale(train_features)
    test_features_prep = preprocessing.scale(test_features)
    SVM.fit(train_features_prep, train_labels)
    y_pred = SVM.predict(test_features_prep)
    print(f"SVM: {get_roc(test_labels, y_pred)}")

    # Logistic Regression
    Logistic.fit(train_features, train_labels)
    y_pred = Logistic.predict(test_features)
    print(f"Logistic: {get_roc(test_labels, y_pred)}")

    # RandomForest
    RandomForest.fit(train_features, train_labels)
    y_pred = RandomForest.predict(test_features)
    print(f"RandomForest: {get_roc(test_labels, y_pred)}")

    # AdaBoost
    AdaBoost.fit(train_features, train_labels)
    y_pred = AdaBoost.predict(test_features)
    print(f"AdaBoost: {get_roc(test_labels, y_pred)}")

    # XGBoost
    xgb.fit(train_features, train_labels)
    y_pred = xgb.predict(test_features)
    print(f"XGB: {get_roc(test_labels, y_pred)}")

    # Extreme Randomized Classifier
    xtra.fit(train_features, train_labels)
    y_pred = xtra.predict(test_features)
    print(f"Xtreme Random Forest: {get_roc(test_labels, y_pred)}")


def importance_xgb(model, test_features, test_labels):
    """Understand predictive power of different features for the XGBoost trained model"""
    feature_names = [
        "Average venue visists",
        "Average venue sentiment",
        "Venue degree",
        "User degree",
        "Adamic/Adar (users)",
        "Adamic/Adar (venues)",
        "Jaccard (venues)",
        "Tag overlap",
    ]

    r = permutation_importance(model, test_features, test_labels, n_repeats=10)

    for i in r.importances_mean.argsort()[::-1]:
        print(
            f"{feature_names[i]:<8}: "
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}"
        )
