from math import sqrt
import time

import matplotlib.pyplot as plt
import networkx as nx


def neighborhood_size(node, neighborhoods):
    return len(neighborhoods[node]) + 1

# structural similarity
def structural_similarity(node_1, node_2, neighborhoods):
    if node_1 not in neighborhoods or node_2 not in neighborhoods:
        return -1
    else:
        return len(neighborhoods[node_1] & neighborhoods[node_2]) / \
               sqrt(neighborhood_size(node_1, neighborhoods) * neighborhood_size(node_2, neighborhoods))

# maxumim structural similarity of a pair in a circle
def max_structural_similarity(circle, neighborhoods, graph):
    return max(similarities(circle, neighborhoods, graph))

# average structural similarity of pairs in a circle
def avg_structural_similarity(circle, neighborhoods, graph):
    s = [sim for sim in similarities(circle, neighborhoods, graph) if sim >= 0]
    if s:
        return sum(s)/float(len(s))
    else:
        return -1

# return a list of similarity values computed for pairs of neighbors in a circle
def similarities(circle, neighborhoods, graph):
    result = []
    for node_1 in circle:
        for node_2 in circle:
            if node_1 != node_2 and ( (node_1, node_2) in graph.edges() or (node_2, node_1 in graph.edges) ):
                result.append(structural_similarity(node_1, node_2, neighborhoods))
    return result

# circle similarity measure
def circle_similarity(circle_1, circle_2):
    if not circle_1 or not circle_2:
        return 0
    else:
        #Jaccard similarity
        return len(circle_1 & circle_2) / float(len(circle_1 | circle_2))

# find best match for circle, return clusterID
def match_circle(circle, clusters):
    clusterID, similarity = max([(cl, circle_similarity(circle, clusters[cl])) for cl in clusters], key=lambda x: x[1])
    if similarity > 0:
        return (clusterID, similarity)
    else:
        return (None, -1)

# returns similarity score for the clustering
def evaluate_clustering(circles, clusters):
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

# adjusted rand index
def ari(clusters_x, clusters_y):
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

# display cluster size distribution for the given clustering
def plot_cluster_size_distribution(clusters):
    siz = [len(clusters[cl]) for cl in clusters]
    x = sorted(list(set(siz)))
    y = [siz.count(i) for i in x]
    plt.xlabel('Cluster size')
    plt.ylabel('Count')
    plt.plot(x, y, 'co')
    plt.show()

# display cluster size histogram for the given clustering
def hist_cluster_size(clusters, bucket=50):
    siz = [len(clusters[cl]) for cl in clusters]
    plt.xlabel('Cluster size')
    plt.ylabel('Count')
    plt.hist(siz, bucket , color='c')
    plt.show()

# measure runtime of algorithm, assumes the received object has a run method
def runtime(alg_instance):
    start = time.clock()
    alg_instance.run()
    end = time.clock()
    return end - start
