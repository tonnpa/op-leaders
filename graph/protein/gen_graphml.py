import networkx as nx

file = 'protein_interaction.edges'

graph = nx.Graph()
with open(file, 'r') as edge_list:
    for edge in edge_list.readlines():
        from_node, to_node = edge.strip().split(' ')
        graph.add_edge(int(from_node), int(to_node))

nx.write_graphml(graph, 'protein.graphml')