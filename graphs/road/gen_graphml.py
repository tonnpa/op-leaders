import networkx as nx

file = 'san_francisco.edges'

graph = nx.Graph()
with open(file, 'r') as edge_list:
    for edge in edge_list.readlines():
        _, from_node, to_node, _ = edge.strip().split(' ')
        graph.add_edge(int(from_node), int(to_node))

nx.write_graphml(graph, 'road.graphml')
