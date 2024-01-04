import networkx as nx
import numpy as np


def two_hops_set(graph, node):
    """
    Get a set of nodes two hops away from a given node
    """
    two_hop = []
    one_hop = list(graph.neighbors(node))
    for neighbor in one_hop:
        two_hop_neighbors = list(graph.neighbors(neighbor))
        two_hop += two_hop_neighbors
    return set(two_hop)


def one_hop_set(graph, node):
    """
    Get a set of nodes one hop away from a given node
    """
    return set(graph.neighbors(node))


def common_neighbors(graph, node_a, node_b):
    """
    Calculate common neighbors for a node pair in a bipartite graph
    """
    same_nodes = two_hops_set(graph, node_a)
    opposite_nodes = one_hop_set(graph, node_b)
    cn = len(same_nodes.intersection(opposite_nodes))
    return cn


def common_neighbors_pred(graph, examples, users=True):
    """
    Calculate common neighbors for a set of node pairs in a bipartite graph
    """
    cn = []
    for edge in examples:
        if users:
            same_nodes = two_hops_set(graph, edge[0])
            opposite_nodes = one_hop_set(graph, edge[1])
        else:
            same_nodes = two_hops_set(graph, edge[1])
            opposite_nodes = one_hop_set(graph, edge[0])
        cn.append(len(same_nodes.intersection(opposite_nodes)))
    return cn


def jaccard(graph, node_a, node_b):
    """
    Calculate Jaccard's coefficient for a node pair in a bipartite graph
    """
    same_nodes = two_hops_set(graph, node_a)
    opposite_nodes = one_hop_set(graph, node_b)
    inter_size = len(same_nodes.intersection(opposite_nodes))
    union_size = len(same_nodes.union(opposite_nodes))
    if union_size > 0:
        jacc = inter_size / union_size
        return jacc
    else:
        return 0


def jaccard_pred(graph, examples, users=True):
    """
    Calculate Jaccard's coefficient for a set of node pairs in a bipartite graph
    """
    jacc = []
    for edge in examples:
        if users:
            same_nodes = two_hops_set(graph, edge[0])
            opposite_nodes = one_hop_set(graph, edge[1])
        else:
            same_nodes = two_hops_set(graph, edge[1])
            opposite_nodes = one_hop_set(graph, edge[0])
        inter_size = len(same_nodes.intersection(opposite_nodes))
        union_size = len(same_nodes.union(opposite_nodes))
        jacc.append(inter_size / union_size)

    return jacc


def adamic_adar(graph, node_a, node_b):
    """
    Calculate Adamic/Adar index for a node pair in a bipartite graph
    """
    same_nodes = two_hops_set(graph, node_a)
    opposite_nodes = one_hop_set(graph, node_b)
    inter_list = same_nodes.intersection(opposite_nodes)
    ad_ar = 0
    for node in inter_list:
        deg = graph.degree(node)
        if deg > 1:
            ad_ar += 1 / np.log(deg)
        else:
            ad_ar += 0
    return ad_ar


def adamic_adar_pred(graph, examples, users=True):
    """
    Calculate Adamic/Adar index for a set of node pairs in a bipartite graph
    """
    ad_ar_all = []

    for edge in examples:
        if users:
            same_nodes = two_hops_set(graph, edge[0])
            opposite_nodes = one_hop_set(graph, edge[1])
            inter_list = same_nodes.intersection(opposite_nodes)
        else:
            same_nodes = two_hops_set(graph, edge[1])
            opposite_nodes = one_hop_set(graph, edge[0])
            inter_list = same_nodes.intersection(opposite_nodes)
        ad_ar = 0
        for node in inter_list:
            deg = graph.degree(node)
            if deg > 1:
                ad_ar += 1 / np.log(deg)
            else:
                ad_ar += 0
        ad_ar_all.append(ad_ar)

    return ad_ar_all


def get_gcc(graph):
    """
    Create the greatest connected component for a graph and print its statistics
    """
    # Find the number of connected component
    num_of_cc = nx.number_connected_components(graph)
    print("Number of connected components: {}".format(num_of_cc))

    # Get the greatest connected component subgraph
    gcc_nodes = max(nx.connected_components(graph), key=len)
    gcc = graph.subgraph(gcc_nodes)
    node_fraction = gcc.number_of_nodes() / float(graph.number_of_nodes())
    edge_fraction = gcc.number_of_edges() / float(graph.number_of_edges())

    print("Fraction of nodes in GCC: {:.3f}".format(node_fraction))
    print("Fraction of edges in GCC: {:.3f}".format(edge_fraction))

    return gcc
