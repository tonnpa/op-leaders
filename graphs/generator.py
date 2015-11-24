import argparse
import os
import re

import networkx as nx

parser = argparse.ArgumentParser()
# input
parser.add_argument("-e", "--edge_list", type=str)
parser.add_argument("-n", "--ego_networks", type=str)

parser.add_argument("-o", "--output_file", type=str, required=True)
args = parser.parse_args()

graph = nx.Graph()

#
if args.edge_list:
    file = args.edge_list
    assert os.path.isfile(file)
    with open(file, 'r') as edge_list:
        for edge in edge_list.readlines():
            _, from_node, to_node, _ = edge.strip().split(' ')
            graph.add_edge(int(from_node), int(to_node))

    nx.write_graphml(graph, args.output_file)

if args.ego_networks:
    dir = args.ego_networks
    assert os.path.isdir(dir)
    files = os.listdir(dir)
    for file in files:
        with open(dir + file, 'r') as egonet:
            for line in egonet.readlines():
                from_node, to_node_list = line.strip().split(':')
                to_nodes = re.findall("\d+", to_node_list)

                for n in to_nodes:
                    graph.add_edge(int(from_node), int(n))

    nx.write_graphml(graph, args.output_file)
