# coding=utf-8

from math import sqrt

__author__ = 'tonnpa'


class SCAN:

    # categorization
    member       = 'member'
    non_member   = 'non-member'
    unclassified = 'unclassified'

    def __init__(self, graph, epsilon=0.7, mu=2):
        self.graph   = graph
        self.epsilon = epsilon
        self.mu      = mu

        # set of structure reachable, but not core nodes
        self.hub     = set()
        self.outlier = set()

        # set all node labels to unclassified
        self.labels = dict((n, SCAN.unclassified) for n in self.graph.nodes())

        # set clusterID assignments to empty
        self.clusterID = dict((n, set()) for n in self.graph.nodes())

        # epsilon neighborhood of nodes
        self.e_neighborhood = dict((n, self._get_e_neighborhood(n)) for n in self.graph.nodes())

    def _get_e_neighborhood(self, node):
        e_neighborhood = set()
        neighborhood = self.neighborhood(node)
        for n in neighborhood:
            sigma = self.sigma(node, n)
            if sigma >= self.epsilon:
                e_neighborhood.add(n)
                # print('sigma({0},{1}: {2}'.format(node, n, sigma))
        return e_neighborhood

    def _classify(self, node, label, cluster=None):
        self.labels[node] = label
        if cluster:
            self.clusterID[node].add(cluster)

    def _classify_nonmembers(self):
        non_members = (n for n in self.graph.nodes() if self.is_nonmember(n))
        for node in non_members:
            classes = set()
            for n in self.graph.neighbors(node):
                classes = classes | self.clusterID[n]
            if len(classes) >= 2:
                self.hub.add(node)
            else:
                self.outlier.add(node)

    def run(self):
        cluster_id = 0

        for node in self.graph:
            if self.is_unclassified(node):
                if self.is_core(node):
                    # generate new cluster ID
                    cluster_id += 1
                    # classify node
                    self._classify(node, SCAN.member, cluster_id)
                    # insert all x ∈ Nε(node) into queue
                    queue = list(self.eneighborhood(node) - {node})
                    while queue:
                        y = queue.pop()
                        # classify
                        self._classify(y, SCAN.member, cluster_id)
                        if self.is_core(y):
                            for x in (self.eneighborhood(y) - {y}):
                                if not self.is_member(x) and x not in queue:
                                    # add neighboring nodes to queue for later classification
                                    queue.append(x)
                else:
                    # label node as non-member
                    self._classify(node, SCAN.non_member)

        # classify non-member nodes
        self._classify_nonmembers()

        # save number of clusters
        self.cluster_cnt = cluster_id

    def colors(self):
        colors = []
        for node in self.graph.nodes():
            if node in self.hub:
                colors.append(self.cluster_cnt + 1)
            elif node in self.outlier:
                colors.append(0)
            else:
                colors.append(next(iter(self.clusterID[node])))

        return colors

    def is_core(self, node):
        return len(self.eneighborhood(node)) >= self.mu

    def is_member(self, node):
        return self.labels[node] == SCAN.member

    def is_nonmember(self, node):
        return self.labels[node] == SCAN.non_member

    def is_unclassified(self, node):
        return self.labels[node] == SCAN.unclassified

    def neighborhood(self, node):
        nh = set(self.graph.neighbors(node))
        nh.add(node)
        return nh

    def eneighborhood(self, node):
        return self.e_neighborhood[node]

    def sigma(self, n1, n2):
        common_neighbors = self.neighborhood(n1) & self.neighborhood(n2)
        return len(common_neighbors) / sqrt(len(self.neighborhood(n1)) * len(self.neighborhood(n2)))

    def direct_reach(self, node):
        reach = self.eneighborhood(node) - {node} if self.is_core(node) else set()
        return reach

    def hubs(self):
        return self.hub

    def outliers(self):
        return self.outlier

    def number_of_clusters(self):
        return self.cluster_cnt
