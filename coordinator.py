import matplotlib.pyplot   as plt
import networkx            as nx

import algorithms.autopart as ap
import algorithms.scan     as sc

__author__ = 'tonnpa'

"""
example graphs adjacency matrix

 0.  1.  1. | 1.  0.  0.  0. | 0.  0. | 0.  0.  1.  0.  1.
 1.  0.  1. | 0.  0.  1.  0. | 0.  0. | 0.  0.  1.  0.  0.
 1.  1.  0. | 1.  0.  0.  0. | 0.  0. | 1.  0.  0.  0.  0.
---------------------------------------
 1.  0.  1. | 0.  0.  0.  0. | 0.  0. | 1.  0.  0.  0.  1.
 0.  0.  0. | 0.  0.  0.  0. | 1.  0. | 0.  1.  0.  1.  0.
 0.  1.  0. | 0.  0.  0.  0. | 0.  0. | 0.  0.  0.  0.  0.
 0.  0.  0. | 0.  0.  0.  0. | 1.  1. | 1.  0.  0.  1.  0.
---------------------------------------
 0.  0.  0. | 0.  1.  0.  1. | 0.  1. | 0.  1.  0.  1.  0.
 0.  0.  0. | 0.  0.  0.  1. | 1.  0. | 1.  1.  0.  0.  0.
---------------------------------------
 0.  0.  1.   1.  0.  0.  1.   0.  1.   0.  1.  0.  0.  1.
 0.  0.  0.   0.  1.  0.  0.   1.  1.   1.  0.  0.  0.  0.
 1.  1.  0.   0.  0.  0.  0.   0.  0.   0.  0.  0.  0.  1.
 0.  0.  0.   0.  1.  0.  1.   1.  0.   0.  0.  0.  0.  0.
 1.  0.  0.   1.  0.  0.  0.   0.  0.   1.  0.  1.  0.  0.
"""


def main():
    graphml = 'graphs/examples/polbooks.graphml'
    graph = nx.read_graphml(graphml)
    autopart = ap.Autopart(graph)
    autopart.run()

def run_scan_example():
    graphml = '/home/tonnpa/Documents/datasets/example.graphml'
    graph = nx.read_graphml(graphml, node_type=int)
    scan_obj = sc.SCAN(graph)
    scan_obj.run()
    print('hubs: ', scan_obj.hub)
    print('outliers: ', scan_obj.outlier)
    print('cluster count: ', scan_obj.number_of_clusters())


def run_scan():
    graphml = '/home/tonnpa/Documents/datasets/books/polbooks.graphml'
    # graphml = '/home/tonnpa/Documents/datasets/example.graphml'
    graph  = nx.read_graphml(graphml, node_type=float)
    layout = nx.spring_layout(graph)

    for epsi in (0.4, 0.5, 0.6, 0.7):
        scan_obj = sc.SCAN(graph, epsilon=epsi, mu=3)
        scan_obj.run()
        # print('hubs: ', scan_obj.hub)
        # print('outliers: ', scan_obj.outlier)
        # print('cluster count: ', scan_obj.number_of_clusters())
        # print(sorted(scan_obj.colors()))
        nx.draw_networkx(graph, pos=layout, node_color=scan_obj.colors())
        plt.show()


def test_scan():
    graphml = 'graphs/examples/scan.graphml'
    graph = nx.read_graphml(graphml, node_type=int)
    scan_obj = sc.SCAN(graph)
    cores = set()
    non_members = set()
    for node in scan_obj.graph.nodes():
        if scan_obj.is_core(node):
            cores.add(node)
        else:
            non_members.add(node)
    assert(cores == {3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20})
    assert(non_members == {7, 14})
    assert(scan_obj.eneighborhood(5) == {3, 4, 5, 6, 15, 20})
    assert(scan_obj.sigma(3, 7) - 0.50709255283711 < 0.0001)

    scan_obj.run()
    assert(scan_obj.hubs() == {7})
    assert(scan_obj.outliers() == {14})


def test_autopart():
    from math import ceil, log

    graphml  = 'graphs/examples/scan.graphml'
    graph    = nx.read_graphml(graphml)
    autopart = ap.Autopart(graph)
    autopart.k = 3
    group_0 = [1, 2, 3]
    group_1 = [4, 5, 6, 7]
    group_2 = [8, 9]
    autopart.map_g_n = {0: group_0, 1: group_1, 2: group_2}

    assert(autopart.group_size(0) == len(group_0))
    assert(autopart.group_size(1) == len(group_1))
    assert(autopart.group_size(2) == len(group_2))

    assert(ap.log_star(16) == 7)
    assert(ap.log2(0.5) == log(0.5, 2))
    assert(ap.log2(20)  == log(20,  2))

    assert(autopart.code_group_sizes() == ceil(log(7, 2)) + ceil(log(4, 2)))
    block_weights = ceil(log(16 + 1, 2)) + ceil(log(12 + 1, 2)) * 2 + \
                    ceil(log( 9 + 1, 2)) + ceil(log( 8 + 1, 2)) * 2 + \
                    ceil(log( 4 + 1, 2)) + ceil(log( 6 + 1, 2)) * 2
    assert (autopart.code_block_weights() == block_weights)

    assert(autopart.group_start_idx(0) == 0)
    assert(autopart.group_start_idx(1) == 3)
    assert(autopart.group_start_idx(2) == 7)

    # autopart._recalculate_block_properties()
    assert(autopart.block_size(0, 0) == 9)
    assert(autopart.block_size(1, 2) == 8)
    assert(autopart.block_size(1, 1) == 16)

    assert(autopart.block_weight(0, 0) == 6)
    assert(autopart.block_weight(2, 1) == 3)

    assert(autopart.block_density(0, 0) == (6.0 + 0.5) / (9 + 1))
    assert(autopart.block_density(2, 1) == (3.0 + 0.5) / (8 + 1))

    assert(autopart.row_weight(0, 0) == 2)
    assert(autopart.row_weight(1, 1) == 1)
    assert(autopart.row_weight(2, 2) == 0)
    assert(autopart.col_weight(0, 2) == 0)
    assert(autopart.col_weight(1, 0) == 2)
    assert(autopart.col_weight(2, 1) == 1)


def write_egonetworks():
    import graphs.portal444.discussion_graph as dg

    src_dir = '/tmp/posts'
    g = dg.build_graph(src_dir, g_path='/tmp/test.graphml')
    dg.write_egonets('/tmp/test.graphml', '/tmp/egonets')

    print(len(g.nodes()))
    print(len(g.edges()))

if __name__ == '__main__':
    main()
