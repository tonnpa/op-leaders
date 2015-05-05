__author__ = 'tonnpa'

import json
import re
import os

import networkx as nx

import input_check as ic
from graph import primitive_graph as pgraph


def build_graph(src_dir, **kwargs):
    # INPUT CHECK
    # ===========
    assert ic.src_dir_exists(src_dir)
    assert not ic.src_dir_empty(src_dir)

    target_file = None
    write_request = False
    if kwargs:
        if 'g_path' in kwargs:
            target_file = kwargs['g_path']
            ic.tgt_file_exists(target_file)
            assert ic.file_extension_match(target_file, '.graphml')
            write_request = True
        else:
            print('ERROR: Invalid keyword arguments supplied')
            return

    files = os.listdir(src_dir)
    files.sort()

    graph = nx.DiGraph()
    comments_authorship = {}
    # Key: comment ID, Value: author username
    authors_linkage = {}
    # Key: sender author username, Value: recipient author username

    # MAIN CYCLE
    # ==========
    for idx, file in enumerate(files):
        if idx % 100 == 0:
            print('Progress: {0:.2%}'.format(float(idx)/len(files)))
        with open(src_dir + '/' + file, 'r') as json_file:
            data = json.loads(json_file.read())

        if re.match('.*_\d\.json$', file) is None:
            # free memory
            comments_authorship.clear()
            authors_linkage.clear()

        # 1. phase
        pgraph.build_adjacency_list(authors_linkage, data['response'], comments_authorship)
        # 2. phase
        pgraph.build_graph(graph, authors_linkage)

    # FAULT DETECTION
    # ==============================
    invalid_edges = [e for e in graph.edges(data=True) if e[2]['createdAt'] > e[2]['stoppedAt']]
    assert len(invalid_edges) == 0
    invalid_nodes = [n for n in graph.nodes(data=True) if n[1]['joinedAt'] > n[1]['inactiveSince']]
    assert len(invalid_nodes) == 0

    if write_request:
        nx.write_graphml(graph, target_file)

    print('INFO: Graph construction complete')
    return graph


def write_egonets(g_path, tgt_dir):
    """
    This function creates a subgraph of each node's egonets in the
    complete graph.

    Egonet definition: the central node C and all other nodes that
    have an edge pointing toward central node C
    All edges between these nodes are kept.

    :param g_path: path to the complete graph
    :param tgt_dir: path to the target folder where all subgraph
           graphml files will be written
    :return:
    """
    # INPUT CHECK
    # ===========
    assert ic.src_file_exists(g_path)
    assert ic.file_extension_match(g_path, '.graphml')
    assert ic.tgt_dir_exists(tgt_dir)

    graph = nx.read_graphml(g_path)

    # MAIN CYCLE
    # ==========
    for idx, node in enumerate(graph.nodes()):
        if idx % 100 == 0:
            print('Progress: {0:.2%}'.format(float(idx)/graph.order()))

        # create list of nodes
        en_nodes = [node]
        en_nodes.extend(graph.predecessors(node))

        # create subgraph
        en_graph = graph.subgraph(en_nodes)

        nx.write_graphml(en_graph, tgt_dir + '/' + str(idx).zfill(5) + '_' + node + '.graphml')