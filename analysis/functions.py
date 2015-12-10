import time
from math import sqrt

import matplotlib.pyplot as plt
import networkx as nx


def neighborhood_size(node, neighborhoods):
    """
    SCAN Section 3.1 Definition 1
    """
    return len(neighborhoods[node]) + 1

def structural_similarity(node_1, node_2, neighborhoods):
    """
    SCAN Section 3.1 Definition 2
    """
    if node_1 not in neighborhoods or node_2 not in neighborhoods:
        return -1
    else:
        return len(neighborhoods[node_1] & neighborhoods[node_2]) / \
               sqrt(neighborhood_size(node_1, neighborhoods) * neighborhood_size(node_2, neighborhoods))

def max_structural_similarity(circle, neighborhoods, graph):
    """
    :return: maximum structural similarity of a pair in a circle
    """
    return max(similarities(circle, neighborhoods, graph))

def avg_structural_similarity(circle, neighborhoods, graph):
    """
    :return: average structural similarity of pairs in a circle
    """
    s = [sim for sim in similarities(circle, neighborhoods, graph) if sim >= 0]
    if s:
        return sum(s)/float(len(s))
    else:
        return -1

def similarities(circle, neighborhoods, graph):
    """
    :return: a list of similarity values computed for pairs of neighbors in a circle
    """
    result = []
    for node_1 in circle:
        for node_2 in circle:
            if node_1 != node_2 and ( (node_1, node_2) in graph.edges() or (node_2, node_1 in graph.edges) ):
                result.append(structural_similarity(node_1, node_2, neighborhoods))
    return result

def circle_similarity(circle_1, circle_2):
    """
    :return: circle similarity measure
    """
    if not circle_1 or not circle_2:
        return 0
    else:
        #Jaccard similarity
        return len(circle_1 & circle_2) / float(len(circle_1 | circle_2))

def match_circle(circle, clusters):
    """
    :return: find best match for circle, return clusterID
    """
    clusterID, similarity = max([(cl, circle_similarity(circle, clusters[cl])) for cl in clusters], key=lambda x: x[1])
    if similarity > 0:
        return (clusterID, similarity)
    else:
        return (None, -1)

def evaluate_clustering(circles, clusters):
    """
    :return: Jaccard similarity score for the clustering
    """
    g = nx.Graph()
    for circle in circles:
        for cluster in clusters:
            similarity = circle_similarity(circles[circle], clusters[cluster])
            if similarity > 0:
                g.add_edge(circle, cluster, weight=similarity)
    match = nx.max_weight_matching(g)

    score = 0
    for circle in circles:
        try:
            score += circle_similarity(circles[circle], clusters[match[circle]])
        except KeyError:
            continue

    return score

def ari(clusters_x, clusters_y):
    """
    :return: adjusted rand index SCAN Section 6.2.1
    """
    def nCr(num):
        return num * (num - 1) / 2.0

    n_x = sum([len(clusters_x[cl]) for cl in clusters_x])
    n_y = sum([len(clusters_y[cl]) for cl in clusters_y])
    assert n_x == n_y
    n = n_x

    sum_i  = sum([nCr(len(clusters_x[cl])) for cl in clusters_x])
    sum_j  = sum([nCr(len(clusters_y[cl])) for cl in clusters_y])
    sum_ij = sum([nCr(len(clusters_x[cl_x] & clusters_y[cl_y])) for cl_x in clusters_x for cl_y in clusters_y])
    expected_idx = sum_i * sum_j / nCr(n)

    return ( sum_ij - expected_idx ) / ( 0.5 * (sum_i + sum_j) - expected_idx )

def plot_cluster_size_distribution(clusters):
    """
    display cluster size distribution for the given clustering
    """
    siz = [len(clusters[cl]) for cl in clusters]
    x = sorted(list(set(siz)))
    y = [siz.count(i) for i in x]
    plt.xlabel('Cluster size')
    plt.ylabel('Count')
    plt.plot(x, y, 'co')
    plt.show()

def hist_cluster_size(clusters, bucket=50):
    """
    display cluster size histogram for the given clustering
    """
    siz = [len(clusters[cl]) for cl in clusters]
    plt.xlabel('Cluster size')
    plt.ylabel('Count')
    plt.hist(siz, bucket , color='c')
    plt.show()

def runtime(function, parameter):
    """
    measure runtime of function
    """
    start = time.clock()
    function(parameter)
    end = time.clock()
    return end - start
