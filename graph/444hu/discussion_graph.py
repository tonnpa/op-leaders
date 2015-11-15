__author__ = 'tonnpa'

import csv
import json
import os
import re
from datetime import date, timedelta

import networkx as nx

import input_check     as ic
import primitive_graph as pgraph


def build_graph(src_dir, **kwargs):
    """
    Builds graph from json files retrieved from disqus listPosts.
    :param src_dir: the directory containing the json posts files
    :param kwargs: if g_path is specified, the result will be saved
                   as a graphml file
    :return: the constructed graph
    """
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


def write_features(src_dir, tgt_file,
                   start_date=date(2014, 1, 1), end_date=date(2015, 1, 1),
                   period_length=92):
    """
    This method creates a single csv file containing all the node
    and edge features of the given graphs.

    Constraints:
    1. graph naming in the source folder:
        \d{5}_(node_name).graphml

    :param src_dir: the folder that contains that graphs to be processed
    :param tgt_file: the output csv file, which follows this structure:
                     node number; node name; node feature; edge feature; period
    :param start_date: start of the inspection interval
    :param end_date: end of the inspection interval
    :param period_length: (measured in days): the length of the subintervals
                          into which the inspection interval is divided
                          e.g. a 92 period length would approximate a quarter of a year
    :return:
    """
    # INPUT CHECK
    # ===========
    assert ic.src_dir_exists(src_dir)
    assert not ic.src_dir_empty(src_dir)
    ic.tgt_file_exists(tgt_file)

    files = os.listdir(src_dir)
    files.sort()

    with open(tgt_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['node_number', 'node_name', 'node_feature', 'edge_feature', 'quarter'])

        # MAIN CYCLE
        # ==========
        for idx, file in enumerate(files):
            if idx % 20 == 0:
                print(file)
            node_number = file[:5]
            node_name = file[6:len(file) - 8]
            subgraph = nx.read_graphml(file)

            # overall
            node_feature = subgraph.order()
            edge_feature = subgraph.size()

            writer.writerow([node_number, node_name, node_feature, edge_feature, 'all'])

            # periods
            from_date = start_date
            to_date = from_date + timedelta(days=period_length)
            period = 1

            while from_date < end_date:
                from_date_str = from_date.isoformat()
                to_date_str = to_date.isoformat()

                sel_nodes = ([n for n in subgraph.nodes(data=True)
                              if (n[1]['joinedAt'] <= to_date_str and
                                  n[1]['inactiveSince'] >= from_date_str)])
                sel_edges = ([e for e in subgraph.edges(data=True)
                              if (e[2]['createdAt'] <= to_date_str and
                                  e[2]['stoppedAt'] >= from_date_str)])

                if node_name in [sn[0] for sn in sel_nodes]:
                    g = nx.DiGraph()
                    g.add_nodes_from(sel_nodes)
                    g.add_edges_from(sel_edges)
                    sen_nodes = [node_name]
                    sen_nodes.extend(g.predecessors(node_name))
                    sg = g.subgraph(sen_nodes)

                    snfeat = sg.order() #subgraph node feature
                    sefeat = sg.size()  #subgraph edge feature

                    # write features to file
                    writer.writerow([node_number, node_name, snfeat, sefeat, period])
                    # clear graphs
                    g.clear()
                    sg.clear()

                from_date = to_date
                to_date = to_date + timedelta(days=period_length)
                period += 1