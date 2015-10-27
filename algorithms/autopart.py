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
        self.map_g_n    = {0: set(self.graph.nodes())}                                # group node mapping
        self.map_n_g    = dict((n, 0) for n in self.graph.nodes())                    # node group mapping
        self.map_n_r    = dict((n, idx) for idx, n in enumerate(self.graph.nodes()))  # node row number mapping
        # cache properties for efficiency
        self._recalculate_block_properties()

    def _block_density(self, group_i, group_j):
        if self.block_size(group_i, group_j) == 0:
            return 0
        else:
            return self.block_weight(group_i, group_j) / self.block_size(group_i, group_j)

    def _block_weight(self, group_i, group_j):
        r_from = self.group_start_idx(group_i)
        r_to   = r_from + self.group_size(group_i)
        c_from = self.group_start_idx(group_j)
        c_to   = c_from + self.group_size(group_j)
        return float(self.adj_matrix[r_from:r_to, c_from:c_to].sum())

    def _add_new_group(self):
        new_group = self.k
        self.map_g_n[new_group] = set()
        self.k += 1
        self._recalculate_block_properties()

    def _move_node_to_new_group(self, node):
        self.map_g_n[self.map_n_g[node]].remove(node)
        self.map_g_n[self.k - 1].add(node)
        self.map_n_g[node] = self.k - 1

        self._rearrange_matrix_and_mappings(self.map_g_n, self.map_n_g)
        self._recalculate_block_properties()

    def _inner_loop(self):
        inner_loop_it = 0
        while True:
            map_n_g = {}
            map_g_n = dict((g, set()) for g in self.groups())
        # STEP 1: assign nodes to node group G_x(t+1)
            for node in self.nodes():
                # the next group is the one with the lowest rearrange cost
                next_grp = min(self.groups(), key=lambda g: self.rearrange_cost(node, self.map_n_g[node], g))
                map_n_g[node] = next_grp
                map_g_n[next_grp].add(node)

            prev_total_cost = self.total_cost()
            self._rearrange_matrix_and_mappings(map_g_n, map_n_g)
        # STEP 2: with respect to G(t+1) recompute the matrices D^t+1_i,j and the corresponding P^t+1_i,j
            self._recalculate_block_properties()
            print 'After inner optimization ', self.map_g_n
        # STEP 3: if there is no decrease in total cost, stop; otherwise proceed to next iteration
            curr_total_cost = self.total_cost()
            # Theorem 1: after each iteration, the code cost decreases or remains the same
            assert curr_total_cost <= prev_total_cost
            if prev_total_cost - curr_total_cost < epsilon:
                # if there is no decrease in total cost, stop
                break
            else:
                # next iteration
                inner_loop_it += 1
                print('Iteration inner ', inner_loop_it)

    def _rearrange_matrix_and_mappings(self, map_g_n, map_n_g):
        # consistency check
        assert(len(map_n_g) == sum([len(map_g_n[g]) for g in map_g_n]))
        # row order with respect to node IDs
        order_node = [node for group in map_g_n for node in map_g_n[group]]
        # row order with respect to previous row numbers
        order_row  = [self.map_n_r[node] for group in map_g_n for node in map_g_n[group]]
        # rewrite the adjacency matrix according to the new grouping
        # 1. switch rows (by creating a new matrix)
        adj_matrix = np.vstack((self.adj_matrix.todense()[row] for row in order_row))
        # 2. switch columns
        adj_matrix[:, :] = adj_matrix[:, order_row]

        self.adj_matrix = sparse.csr_matrix(adj_matrix)
        self.map_g_n    = map_g_n
        self.map_n_g    = map_n_g
        self.map_n_r    = dict((node, idx) for idx, node in enumerate(order_node))

    def _recalculate_block_properties(self):
        # block weights
        self.w = [[self._block_weight(i, j) for j in self.groups()] for i in self.groups()]
        # block densities
        self.P = [[self._block_density(i, j) for j in self.groups()] for i in self.groups()]

    def run(self):
        outer_loop_it = 0
        while True:
            prev_total_cost = self.total_cost()
            # split node group r with maximum entropy per node
            group_r = max(self.groups(), key=lambda g: self.group_entropy_per_node(g))
        # STEP 1: introduce new group, the other half for splitting
            self._add_new_group()
        # STEP 2: construct initial label map
            for node in list((self.map_g_n[group_r])):
                # place the node into the new group if it decreases the per-node entropy of the group
                if self.group_entropy_per_node_exclude(group_r, node) < self.group_entropy_per_node(group_r):
                    self._move_node_to_new_group(node)
            print 'After splitting:', self.map_g_n
        # STEP 3: run the inner loop algorithm
        #     self._inner_loop()
            curr_total_cost = self.total_cost()
            # Theorem 2: On splitting any node group, the code cost either decreases or remains the same.
            print 'prev cost ', prev_total_cost
            print 'curr cost ', curr_total_cost
            assert curr_total_cost <= prev_total_cost
        # STEP 4: if there is no decrease in total cost, stop; otherwise proceed to next iteration
            if prev_total_cost - curr_total_cost < epsilon:
                break
            else:
                outer_loop_it += 1
                print('Iteration outer ', outer_loop_it)

    def block_code_cost(self, group_i, group_j):
        i, j = group_i, group_j
        cost = 0
        cost -= self.block_weight(i, j) * log2(self.block_density(i, j))
        cost -= (self.block_size(i, j) - self.block_weight(i, j)) * log2(1 - self.block_density(i, j))
        return cost

    def code_cost(self):
        print 'code cost ', sum((self.block_code_cost(i, j) for i in self.groups() for j in self.groups()))
        return sum((self.block_code_cost(i, j) for i in self.groups() for j in self.groups()))

    def description_cost(self):
        # number of groups
        cost = log_star(self.k)
        print('log_star ', log_star(self.k), ' k ', self.k)
        # number of nodes in each node group
        cost += self.code_group_sizes()
        print('code group sizes ', self.code_group_sizes())
        # weight of each D_i,j
        cost += self.code_block_weights()
        print('code block weights', self.code_block_weights())
        return cost

    def total_cost(self):
        return self.description_cost() + self.code_cost()

    def rearrange_cost(self, node, curr_group, next_group):
        x = self.map_n_r[node]  # the row number of the node that is to be placed into a group
        i = next_group          # the group into which the node would be placed
        cost = 0                # cost of shifting rows and columns + double counting
        for j in self.groups():
            cost -= self.row_weight(x, j) * log2(self.P[i][j]) + \
                    (self.group_size(j) - self.row_weight(x, j)) * log2(1 - self.P[i][j])
            cost -= self.col_weight(x, j) * log2(self.P[i][j]) + \
                    (self.group_size(j) - self.col_weight(x, j)) * log2(1 - self.P[i][j])
            cost += self.cell(x, x) * \
                    (log2(self.P[i][curr_group]) + log2(self.P[curr_group][i]) - log2(self.P[i][i]))
            cost += (1 - self.cell(x, x)) * \
                    (log2(1 - self.P[i][curr_group]) + log2(1 - self.P[curr_group][i]) - log2(1 - self.P[i][i]))
        return cost

    def code_block_weights(self):
        return sum((ceil(log2(self.block_size(i, j) + 1)) for i in self.groups() for j in self.groups()))

    def code_group_sizes(self):
        if self.k == 1:
            return log2(len(self.nodes()))
        else:
            sizes = sorted([self.group_size(g) for g in self.groups()], reverse=True)

            def a(group_i):
                # 1: our group numbering starts at 0
                val = 1 - self.k + group_i
                for g in range(group_i, self.k):
                    val += sizes[g]
                return val

            res = 0
            for g in range(self.k - 1):
                res += ceil(log2(a(g)))
            return res

    def block_density(self, group_i, group_j):
        return self.P[group_i][group_j]

    def block_size(self, group_i, group_j):
        return self.group_size(group_i) * self.group_size(group_j)

    def block_weight(self, group_i, group_j):
        return self.w[group_i][group_j]

    def cell(self, row, col):
        return float(self.adj_matrix[row, col])

    def col_weight(self, col, group_i):
        r_from = self.group_start_idx(group_i)
        r_to   = r_from + self.group_size(group_i)
        return float(self.adj_matrix[r_from:r_to, col].sum())

    def groups(self):
        return range(self.k)

    def group_entropy_per_node(self, group_i):
        if self.group_size(group_i) == 0:
            return 0
        else:
            entropy = sum((self.block_code_cost(group_i, g) + self.block_code_cost(g, group_i) for g in self.groups()))
            return entropy / self.group_size(group_i)

    def group_entropy_per_node_exclude(self, group_i, node):
        x = self.map_n_r[node]  # excluded row
        entropy = 0
        for j in self.groups():
            n_rj = n_jr = w_rj = w_jr = p_rj = p_jr = 0
            if j == group_i:     # crossing the same group
                if self.group_size(group_i) > 1:
                    n_rj = n_jr = (self.group_size(j) - 1) * (self.group_size(j) - 1)
                    w_rj = w_jr = self.block_weight(j, j) - self.row_weight(x, group_i) - self.col_weight(x, group_i) \
                                    + self.cell(x, x)
                    p_rj = p_jr = w_rj / n_rj
            elif j == self.k - 1:  # crossing the newest group
                if self.group_size(group_i) > 1:
                    n_rj = n_jr = (self.group_size(group_i) - 1) * (self.group_size(j) + 1)
                    w_rj = self.block_weight(group_i, j) - self.row_weight(x, j) + self.col_weight(x, group_i)
                    w_jr = self.block_weight(j, group_i) - self.col_weight(x, j) + self.row_weight(x, group_i)
                    p_rj = w_rj / n_rj
                    p_jr = w_jr / n_jr
            else:                # crossing any other group
                if self.group_size(group_i) > 1 and self.group_size(j) > 0:
                    n_rj = n_jr = (self.group_size(group_i) - 1) * self.group_size(j)
                    w_rj = self.block_weight(group_i, j) - self.row_weight(x, j)
                    w_jr = self.block_weight(j, group_i) - self.col_weight(x, j)
                    p_rj = w_rj / n_rj
                    p_jr = w_jr / n_jr
            entropy -= w_rj * log2(p_rj) + (n_rj - w_rj) * log2(1 - p_rj)
            entropy -= w_jr * log2(p_jr) + (n_jr - w_jr) * log2(1 - p_jr)
        entropy /= self.group_size(group_i) - 1
        return entropy

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
