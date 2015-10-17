__author__ = 'tonnpa'

import networkx as nx
import matplotlib.pyplot as plt

import algorithms.scan as sc


def main():
    file = '/home/tonnpa/Documents/datasets/books/polbooks.graphml'
    # file = '/home/tonnpa/Documents/datasets/example.graphml'
    graph = nx.read_graphml(file)
    scan_obj = sc.SCAN(graph, epsilon=0.4)
    scan_obj.run()
    print('hubs: ', scan_obj.hub)
    print('outliers: ', scan_obj.outlier)
    print('cluster count: ', scan_obj.number_of_clusters())
    print(sorted(scan_obj.draw()))

    nx.draw_networkx(graph, node_color=scan_obj.draw())
    plt.show()
    # run_example()
    # test()


def run_example():
    file = '/home/tonnpa/Documents/datasets/example.graphml'
    graph = nx.read_graphml(file, node_type=int)
    scan_obj = sc.SCAN(graph)
    # test(scan_obj)
    scan_obj.run()
    print('hubs: ', scan_obj.hub)
    print('outliers: ', scan_obj.outlier)
    print('cluster count: ', scan_obj.number_of_clusters())


def test():
    file = '/home/tonnpa/Documents/datasets/example.graphml'
    graph = nx.read_graphml(file, node_type=int)
    scan_obj = sc.SCAN(graph)
    cores = set()
    non_members = set()
    for node in scan_obj.graph.nodes():
        if (scan_obj.is_core(node)):
            cores.add(node)
        else:
            non_members.add(node)
    assert (cores == {3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20})
    assert (non_members == {7, 14})
    assert (scan_obj.eneighborhood(5) == {3, 4, 5, 6, 15, 20})
    assert (scan_obj.sigma(3, 7) - 0.50709255283711 < 0.0001)

    scan_obj.run()
    assert (scan_obj.hubs() == {7})
    assert (scan_obj.outliers() == {14})


if __name__ == '__main__':
    main()

    # from graph import discussion_graph as DG
    #
    # SRC_DIR = '/tmp/posts'
    #
    # g = DG.build_graph(SRC_DIR, g_path='/tmp/test.graphml')
    #
    # DG.write_egonets('/tmp/test.graphml', '/tmp/egonets')
    #
    # print(len(g.nodes()))
    # print(len(g.edges()))
