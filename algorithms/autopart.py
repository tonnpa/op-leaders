from math  import log, ceil
from scipy import sparse
import logging

import networkx as nx
import numpy    as np

__author__ = 'tonnpa'


epsilon = 0.0001

#TODO
# def log2(x):
#     if x < epsilon:
#         return 0
#     else:
#         return log(x, 2)

def log2(x):
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
        # self._recalculate_block_properties()
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    # @property
    # def graph(self):
    #     return self._graph
    #
    # @property
    # def map_g_n(self):
    #     return self._map_g_n
    #
    # @property
    # def map_n_g(self):
    #     return self._map_n_g
    #
    # @property
    # def map_n_r(self):
    #     return self._map_n_r

    #TODO
    # def _block_density(self, group_i, group_j):
    #     if self.block_size(group_i, group_j) == 0:
    #         return 0
    #     else:
    #         return self.block_weight(group_i, group_j) / self.block_size(group_i, group_j)

    def _block_density(self, group_i, group_j):
        return (self.block_weight(group_i, group_j) + 0.5) / (self.block_size(group_i, group_j) + 1)

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
        # self._recalculate_block_properties()

    def _move_node_to_new_group(self, node):
        self.map_g_n[self.map_n_g[node]].remove(node)
        self.map_g_n[self.k - 1].add(node)
        self.map_n_g[node] = self.k - 1

        self._rearrange_matrix_and_mappings(self.map_g_n, self.map_n_g)
        # self._recalculate_block_properties()

    def _inner_loop(self):
        inner_loop_it = 0
        while True:
            map_n_g = {}
            map_g_n = dict((g, set()) for g in self.groups())
        # STEP 1: assign nodes to node group G_x(t+1)
            for node in self.nodes():
                # the next group is the one with the lowest rearrange cost
                next_grp = min(self.groups(), key=lambda g: self.rearrange_cost(node, g))
                map_n_g[node] = next_grp
                map_g_n[next_grp].add(node)
                logging.info('Move node %s from group %d to %d', node, self.map_n_g[node], next_grp)

            prev_code_cost  = self.code_cost()
            prev_total_cost = self.total_cost()
            self._rearrange_matrix_and_mappings(map_g_n, map_n_g)
        # STEP 2: with respect to G(t+1) recompute the matrices D^t+1_i,j and the corresponding P^t+1_i,j
        #     self._recalculate_block_properties()
            logging.info('After inner optimization %s', self.map_g_n)
        # STEP 3: if there is no decrease in total cost, stop; otherwise proceed to next iteration
            curr_code_cost  = self.code_cost()
            curr_total_cost = self.total_cost()
            # Theorem 1: after each iteration, the code cost decreases or remains the same
            print('prev code cost: ', prev_code_cost)
            print('curr code cost: ', curr_code_cost)
            logging.debug("Adjacency matrix\n%s", self.adj_matrix.todense())
            assert curr_code_cost <= prev_code_cost
            if prev_total_cost - curr_total_cost < epsilon:
                # if there is no decrease in total cost, stop
                break
            else:
                # next iteration
                inner_loop_it += 1
                logging.debug('Iteration inner %d', inner_loop_it)

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

    # def _recalculate_block_properties(self):
    #     # block weights
    #     self.w = [[self._block_weight(i, j) for j in self.groups()] for i in self.groups()]
    #     # block densities
    #     self.P = [[self._block_density(i, j) for j in self.groups()] for i in self.groups()]

    def run(self, debug=False):
        outer_loop_it = 0
        while True:
            prev_code_cost  = self.code_cost()
            prev_total_cost = self.total_cost()
            # split node group r with maximum entropy per node
            group_r = max(self.groups(), key=lambda g: self.group_entropy_per_node(g))
        # STEP 1: introduce new group, the other half for splitting
            self._add_new_group()
        # STEP 2: construct initial label map
            prev_grp_entropy = curr_grp_entropy = next_grp_entropy = None
            for node in list((self.map_g_n[group_r])):
                if curr_grp_entropy:
                    prev_grp_entropy = curr_grp_entropy
                curr_grp_entropy = self.group_entropy_per_node(group_r)
                if prev_grp_entropy and prev_grp_entropy < curr_grp_entropy:
                    # the predicted entropy of the group without the node should be equal with
                    # the entropy computed for the group after the move
                    assert next_grp_entropy == curr_grp_entropy
                next_grp_entropy = self.group_entropy_per_node_exclude(group_r, node)
                logging.debug("Node to be moved: %s", node)
                logging.debug("Curr grp entropy per node: %f", curr_grp_entropy)
                logging.debug("Next grp entropy per node: %f", next_grp_entropy)
                # place the node into the new group if it decreases the per-node entropy of the group
                if self.group_entropy_per_node_exclude(group_r, node) < self.group_entropy_per_node(group_r):
                    self._move_node_to_new_group(node)
                    logging.debug("Mapping: %s", self.map_g_n)
            logging.info("After splitting: %s", self.map_g_n)
            logging.debug("Adjacency matrix\n%s", self.adj_matrix.todense())
        # STEP 3: run the inner loop algorithm
            self._inner_loop()
            curr_code_cost  = self.code_cost()
            curr_total_cost = self.total_cost()
            print('outer prev code cost', prev_code_cost)
            print('outer curr code cost', curr_code_cost)
            # Theorem 2: On splitting any node group, the code cost either decreases or remains the same.
            assert curr_code_cost <= prev_code_cost
        # STEP 4: if there is no decrease in total cost, stop; otherwise proceed to next iteration
            if prev_total_cost - curr_total_cost< epsilon:
                break
            else:
                outer_loop_it += 1
                logging.debug("Iteration outer %d", outer_loop_it)

    def block_code_cost(self, group_i, group_j):
        i, j = group_i, group_j
        cost = 0
        cost -= self.block_weight(i, j) * log2(self.block_density(i, j))
        cost -= (self.block_size(i, j) - self.block_weight(i, j)) * log2(1 - self.block_density(i, j))
        return cost

    def code_cost(self):
        return sum((self.block_code_cost(i, j) for i in self.groups() for j in self.groups()))

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

    def rearrange_cost(self, node, next_group):
        g = self.map_n_g[node]  # the group the node currently belongs to
        x = self.map_n_r[node]  # the row number of the node that is to be placed into a group
        i = next_group          # the group into which the node would be placed
        cost = 0                # cost of shifting rows and columns + double counting
        # for j in self.groups():
        #     cost -= self.row_weight(x, j) * log2(self.P[i][j]) + \
        #             (self.group_size(j) - self.row_weight(x, j)) * log2(1 - self.P[i][j])
        #     cost -= self.col_weight(x, j) * log2(self.P[j][i]) + \
        #             (self.group_size(j) - self.col_weight(x, j)) * log2(1 - self.P[j][i])
        # cost += self.cell(x, x) * \
        #         (log2(self.P[i][g]) + log2(self.P[g][i]) - log2(self.P[i][i]))
        # cost += (1 - self.cell(x, x)) * \
        #         (log2(1 - self.P[i][g]) + log2(1 - self.P[g][i]) - log2(1 - self.P[i][i]))
        # return cost
        for j in self.groups():
            # row = self.row_weight(x, j) * log2(self.block_density(i, j)) + \
            #         (self.group_size(j) - self.row_weight(x, j)) * log2(1 - self.block_density(i, j))
            # print('row ', row)
            # col = self.col_weight(x, j) * log2(self.block_density(j, i)) + \
            #         (self.group_size(j) - self.col_weight(x, j)) * log2(1 - self.block_density(j, i))
            # print('col ', col)
            cost -= self.row_weight(x, j) * log2(self.block_density(i, j)) + \
                    (self.group_size(j) - self.row_weight(x, j)) * log2(1 - self.block_density(i, j))
            cost -= self.col_weight(x, j) * log2(self.block_density(j, i)) + \
                    (self.group_size(j) - self.col_weight(x, j)) * log2(1 - self.block_density(j, i))

        # dxx = self.cell(x, x) * \
        #     (log2(self.block_density(i, g)) + log2(self.block_density(g, i)) - log2(self.block_density(i, i)))
        # print('dxx ', dxx)
        #
        # d1xx = (1 - self.cell(x, x)) * \
        #     (log2(1 - self.block_density(i, g)) + log2(1 - self.block_density(g, i)) - log2(1 - self.block_density(i, i)))
        # print('1-dxx', d1xx)

        # cost += self.cell(x, x) * \
        #     (log2(self.block_density(i, g)) + log2(self.block_density(g, i)) - log2(self.block_density(i, i)))
        # cost += (1 - self.cell(x, x)) * \
        #     (log2(1 - self.block_density(i, g)) + log2(1 - self.block_density(g, i)) - log2(1 - self.block_density(i, i)))

        print('Node ', node, ' from ', g, ' to ', next_group, ' cost ', cost)
        return cost

    def code_block_weights(self):
        return sum((ceil(log2(self.block_size(i, j) + 1)) for i in self.groups() for j in self.groups()))

    def code_group_sizes(self):
        if self.k == 1:
            # return ceil(log2(len(self.nodes())))
            return 0
        else:
            sizes = sorted([self.group_size(grp) for grp in self.groups()], reverse=True)

            def a(group_i):
                # 1: our group numbering starts at 0
                val = 1 - self.k + group_i
                for g in range(group_i, self.k):
                    val += sizes[g]
                return val

            res = 0
            for grp in range(self.k - 1):
                res += ceil(log2(a(grp)))
            return res

    def block_density(self, group_i, group_j):
        return self._block_density(group_i, group_j)
        # return self.P[group_i][group_j]

    def block_size(self, group_i, group_j):
        return self.group_size(group_i) * self.group_size(group_j)

    def block_weight(self, group_i, group_j):
        return self._block_weight(group_i, group_j)
        # return self.w[group_i][group_j]

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
        # assuming when exclusion of a node yields an empty group, the entropy of that is null
        if self.group_size(group_i) == 1 or self.group_size(group_i) == 0:
            return 0

        def P(w, n):
            return (w + 0.5) / (n + 1)

        x = self.map_n_r[node]  # excluded row
        entropy = 0
        for j in self.groups():
            n_rj = n_jr = w_rj = w_jr = p_rj = p_jr = 0
            if j == group_i:     # crossing the same group
                if self.group_size(group_i) > 1:
                    n_rj = n_jr = (self.group_size(j) - 1) * (self.group_size(j) - 1)
                    w_rj = w_jr = self.block_weight(j, j) - self.row_weight(x, group_i) - self.col_weight(x, group_i) \
                                    + self.cell(x, x)
                    p_rj = p_jr = P(w_rj, n_rj)
            elif j == self.k - 1:  # crossing the newest group
                if self.group_size(group_i) > 1:
                    n_rj = n_jr = (self.group_size(group_i) - 1) * (self.group_size(j) + 1)
                    w_rj = self.block_weight(group_i, j) - self.row_weight(x, j) + self.col_weight(x, group_i)
                    w_jr = self.block_weight(j, group_i) - self.col_weight(x, j) + self.row_weight(x, group_i)
                    p_rj = P(w_rj, n_rj)
                    p_jr = P(w_jr, n_jr)
            else:                # crossing any other group
                if self.group_size(group_i) > 1 and self.group_size(j) > 0:
                    n_rj = n_jr = (self.group_size(group_i) - 1) * self.group_size(j)
                    w_rj = self.block_weight(group_i, j) - self.row_weight(x, j)
                    w_jr = self.block_weight(j, group_i) - self.col_weight(x, j)
                    p_rj = P(w_rj, n_rj)
                    p_jr = P(w_jr, n_jr)
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
