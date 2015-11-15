import os
import re

import networkx as nx

data_dir = 'egonets/'
files = os.listdir(data_dir)

graph = nx.Graph()
for file in files:
    with open(data_dir + file, 'r') as egonet:
        for line in egonet.readlines():
            from_node, to_node_list = line.strip().split(':')
            to_nodes = re.findall("\d+", to_node_list)

            for n in to_nodes:
                graph.add_edge(int(from_node), int(n))

nx.write_graphml(graph, 'facebook.graphml')