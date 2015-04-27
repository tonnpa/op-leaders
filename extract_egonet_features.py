#!/usr/bin/env python3
"""
This script creates a single csv file containing all the node
and edge features of the given graphs.

SOURCE_DIR: the folder that contains that graphs to be processed
TARGET_FILE: the output csv file, which follows this structure:
    node number; node name; node feature; edge feature; period

How to specify the period:
    1. choose a start date (e.g. 2014-01-01)
    2. choose an end date (e.g. 2015-01-01)
    3. choose the length of the period in days
        e.g. a 92 period length would approximate a quarter of a year

Constraints:
    1. graph naming in the source folder:
        \d{5}_(node_name).graphml
"""

__author__ = 'tonnpa'

from datetime import date, timedelta
import networkx as nx
import csv
import os

SOURCE_DIR = '/media/sf_Ubuntu/opleaders/egonets_followers'
TARGET_FILE = '/media/sf_Ubuntu/opleaders/egonet_features.csv'

START_DATE = date(2014, 1, 1)
END_DATE = date(2015, 1, 1)
PERIOD_LENGTH = 92

files = os.listdir(SOURCE_DIR)
files.sort()
os.chdir(SOURCE_DIR)

with open(TARGET_FILE, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(['node_number', 'node_name', 'node_feature', 'edge_feature', 'quarter'])

    for idx, file in enumerate(files):
        if idx % 20 == 0:
            print(file)
        nnum = file[:5]
        nname = file[6:len(file) - 8]
        subgraph = nx.read_graphml(file)

        # overall
        nfeat = subgraph.order()
        efeat = subgraph.size()

        writer.writerow([nnum, nname, nfeat, efeat, 'all'])

        # periods
        from_date = START_DATE
        to_date = from_date + timedelta(days=PERIOD_LENGTH)
        period = 1

        while from_date < END_DATE:
            from_date_str = from_date.isoformat()
            to_date_str = to_date.isoformat()

            sel_nodes = ([n for n in subgraph.nodes(data=True)
                          if (n[1]['joinedAt'] <= to_date_str and
                              n[1]['inactiveSince'] >= from_date_str)])
            sel_edges = ([e for e in subgraph.edges(data=True)
                          if (e[2]['createdAt'] <= to_date_str and
                              e[2]['stoppedAt'] >= from_date_str)])

            if nname in [sn[0] for sn in sel_nodes]:
                g = nx.DiGraph()
                g.add_nodes_from(sel_nodes)
                g.add_edges_from(sel_edges)
                sen_nodes = [nname]
                sen_nodes.extend(g.predecessors(nname))
                sg = g.subgraph(sen_nodes)

                snfeat = sg.order()
                sefeat = sg.size()

                # write features to file
                writer.writerow([nnum, nname, snfeat, sefeat, period])
                # clear graphs
                g.clear()
                sg.clear()

            from_date = to_date
            to_date = to_date + timedelta(days=PERIOD_LENGTH)
            period += 1