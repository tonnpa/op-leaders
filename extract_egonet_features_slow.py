__author__ = 'tonnpa'

import networkx as nx
import csv

GRAPHML_PATH = '/media/sf_Ubuntu/opleaders/p_graph.graphml'
# GRAPHML_PATH = '/media/sf_Ubuntu/d_graph_ex.graphml'


g = nx.read_graphml(GRAPHML_PATH)

egonetworks = {}
edge_feature = {}

for node in g.nodes():
    egonetworks[node] = g.predecessors(node)
    edge_feature[node] = 0

# for each node N
for node in g.nodes():
    # for each of N's edges
    for to_node in list(g.successors(node)):
        edge_feature[to_node] += 1

        # check whether any of neighbor node's egonetwork in N's egonetwork contains the edge
        for neighbor_node in g.successors(node):
            if to_node in egonetworks[neighbor_node]:
                edge_feature[neighbor_node] += 1

with open('/tmp/features.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')

    for egonet in egonetworks:
        writer.writerow([egonet, g.in_degree(egonet)+1, edge_feature[egonet]])