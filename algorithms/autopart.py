from copy  import deepcopy
from math  import log, ceil
from scipy import sparse

import networkx as nx
import numpy    as np

__author__ = 'tonnpa'


epsilon = 0.0001


def log2(x):
    if x < epsilon:
        return 0
    else:
        return log(x, 2)


def log_star(x):
    res = 0
    val = log(x, 2)
    while val > epsilon:
        res += val
        val = log(val, 2)
    return res


class Autopart:

    def __init__(self, graph):
        self.graph      = graph
        self.adj_matrix = nx.adjacency_matrix(self.graph, graph.nodes())
        self.k          = 1     # number of groups
        # arbitrary G(0) mapping nodes into k node groups
        # group numbering begins at 0
        self.map_g_n    = {0: self.graph.nodes()}                     # group node mapping
        self.map_n_g    = dict((n, 0) for n in self.graph.nodes())    # node group mapping
        self.map_n_r    = dict((n, idx) for idx, n in enumerate(self.graph.nodes())) # node row number mapping
        # cache properties for efficiency
        self._recalculate_block_properties()

    def _block_weight(self, group_i, group_j):
        r_from = self.group_start_idx(group_i)
        r_to   = r_from + self.group_size(group_i)
        c_from = self.group_start_idx(group_j)
        c_to   = c_from + self.group_size(group_j)
        return float(self.adj_matrix[r_from:r_to, c_from:c_to].sum())

    def _recalculate_block_properties(self):
        # block weights
        self.w = [[self._block_weight(i, j) for j in self.groups()] for i in self.groups()]
        # block densities
        self.P = deepcopy(self.w)
        for i in self.groups():
            for j in self.groups():
                try:
                    self.P[i][j] /= self.group_size(i) * self.group_size(j)
                # empty groups are possible
                except ZeroDivisionError:
                    print self.P[i][j], i, j, self.group_size(i), self.group_size(j)

    def run(self):
        # inner loop
        inner_loop_it = 0
        while True:
            # assign nodes to node group G_x(t+1)
            map_n_g = {}
            map_g_n = dict((g,set()) for g in self.groups())
            for node in self.nodes():
                group    = self.map_n_g[node]
                x        = self.map_n_r[node]
                min_cost = float('inf')
                next_grp  = None
                for i in self.groups():
                    cost = 0
                    for j in self.groups():
                        cost -= self.row_weight(x, j)*log2(self.P[i][j]) + \
                                (self.group_size(j)-self.row_weight(x, j))*log2(1-self.P[i][j])
                        cost -= self.col_weight(x, j)*log2(self.P[i][j]) + \
                                (self.group_size(j)-self.col_weight(x, j))*log2(1-self.P[i][j])
                        cost += self.cell(x, x) * \
                                (log2(self.P[i][group]) + log2(self.P[group][i]) - log2(self.P[i][i]))
                        cost += (1-self.cell(x, x)) * \
                                (log2(1-self.P[i][group]) + log2(1-self.P[group][i]) - log2(1-self.P[i][i]))
                    if cost < min_cost:
                        min_cost = cost
                        next_grp = i
                map_n_g[node] = next_grp
                map_g_n[next_grp].add(node)
            # assert map_n_g and map_g_n are consistent
            assert(len(map_n_g) == sum([len(map_g_n[g]) for g in map_g_n]))
            # rewrite the adjacency matrix according the new grouping
            # row order with respect to node numbers
            order_node = [node for group in map_g_n for node in map_g_n[group]]
            # row order with respect to previous row numbers
            order_row  = [self.map_n_r[node] for group in map_g_n for node in map_g_n[group]]
            # switch rows (by creating a new matrix)
            adj_matrix = np.vstack((self.adj_matrix.todense()[row] for row in order_row))
            # switch columns
            adj_matrix[:,:] = adj_matrix[:,order_row]
            map_n_r = dict((node, idx) for idx, node in enumerate(order_node))
            # with respect to G(t+1) recompute the matrices D^t+1_i,j and
            # the corresponding P^t+1_i,j
            prev_total_cost = self.total_cost()

            self.adj_matrix = sparse.csr_matrix(adj_matrix)
            self.map_g_n    = map_g_n
            self.map_n_g    = map_n_g
            self.map_n_r    = map_n_r

            self._recalculate_block_properties()
            curr_total_cost = self.total_cost()
            # Theorem 1: after each iteration, the code cost decreases or remains the same
            assert curr_total_cost <= prev_total_cost
            if prev_total_cost - curr_total_cost < epsilon:
                # if there is no decrease in total cost, stop
                break
            else:
                # next iteration
                inner_loop_it += 1
                print('Iteration ', inner_loop_it)
        print(self.adj_matrix.todense())

        print sorted([ (n,r) for n, r in self.map_n_r.items() ], key=lambda tuple: tuple[1])

    def code_cost(self):
        cost = 0
        for i in self.groups():
            for j in self.groups():
                n_ij  = self.block_size(i, j)
                cost -= self.w[i][j]*log2(self.P[i][j]) + \
                        (n_ij-self.w[i][j])*log2(1-self.P[i][j])
        return cost

    def description_cost(self):
        # number of groups
        cost = log_star(self.k)
        # number of nodes in each node group
        cost += self.code_group_sizes()
        # weight of each D_i,j
        cost += self.code_block_weights()
        return cost

    def total_cost(self):
        return self.description_cost() + self.code_cost()

    def code_block_weights(self):
        val = 0
        for i in self.groups():
            for j in self.groups():
                val += ceil(log2(self.group_size(i)*self.group_size(j)+1))
        return val

    def code_group_sizes(self):
        sizes = sorted([self.group_size(g) for g in self.groups()], reverse=True)

        def a(group_i):
            # 1: out group numbering starts at 0
            val = 1 - self.k + group_i
            for g in range(group_i, self.k):
                val += sizes[g]
            return val

        val = 0
        for g in range(self.k-1):
            val += ceil(log2(a(g)))
        return val

    def block_density(self, group_i, group_j):
        return self.P[group_i][group_j]

    def block_size(self, group_i, group_j):
        return self.group_size(group_i) * self.group_size(group_j)

    def block_weight(self, group_i, group_j):
        return self.w[group_i][group_j]

    def cell(self, row, col):
        return float(self.adj_matrix[row,col])

    def col_weight(self, col, group_i):
        r_from = self.group_start_idx(group_i)
        r_to   = r_from + self.group_size(group_i)
        return float(self.adj_matrix[r_from:r_to, col].sum())

    def groups(self):
        return range(self.k)

    def group_size(self, group_i):
        return len(self.map_g_n[group_i])

    def group_start_idx(self, group_i):
        return sum([self.group_size(g) for g in range(group_i)])

    def nodes(self):
        return self.graph.nodes()

    def row_weight(self, row, group_i):
        c_from = self.group_start_idx(group_i)
        c_to   = c_from + self.group_size(group_i)
        return float(self.adj_matrix[row, c_from:c_to].sum())


