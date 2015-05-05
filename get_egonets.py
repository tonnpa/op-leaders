#!/usr/bin/env python3
"""
This script creates a subgraph of each node's egonets in the
complete graph.

Egonet definition: the central node C and all other nodes that
    have an edge pointing toward central node C
    All edges between these nodes are kept.

COMP_GRAPH_PATH: path to the complete graph
TARGET_DIR_PATH: path to the target folder where all subgraph
    graphml files will be written
"""

__author__ = 'tonnpa'

import networkx as nx

COMP_GRAPH_PATH = '/media/sf_Ubuntu/opleaders/p_graph.graphml'
TARGET_DIR_PATH = '/media/sf_Ubuntu/opleaders/egonets_followers/'

complete_graph = nx.read_graphml(COMP_GRAPH_PATH)

for idx, node in enumerate(complete_graph.nodes()):
    if idx % 40 == 0:
        print(node + ' ' + str(idx))

    # create list of nodes
    en_nodes = [node]
    en_nodes.extend(complete_graph.predecessors(node))

    # create subgraph
    en_graph = complete_graph.subgraph(en_nodes)

    nx.write_graphml(en_graph, TARGET_DIR_PATH + str(idx).zfill(5) + '_' + node + '.graphml')
