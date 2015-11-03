from math import fabs, exp, log, pow

import numpy as np

__author__ = 'tonnpa'

class Oddball:

    def __init__(self, graph):
        self._graph       = graph
        self._egonetworks = dict((node, {}) for node in graph.nodes())
        self._outlierness = []
        self._theta       = None
        self._C           = None

    @property
    def egonetworks(self):
        return self._egonetworks

    @property
    def graph(self):
        return self._graph

    @property
    def outlierness(self):
        return self._outlierness

    def run(self):
        for node in self.nodes():
            # extract egonetwork
            egonetwork = self.graph.subgraph(self.graph.neighbors(node) + [node])
            self._egonetworks[node]['nodes'] = len(egonetwork.nodes())  # number of nodes in egonet
            self._egonetworks[node]['edges'] = len(egonetwork.edges())  # number of edges in egonet
        # compute power-law coefficients m, C
        nodes = [log(self.n_count(n)    ) for n in self.nodes()]
        edges = [log(self.e_count(n) + 1) for n in self.nodes()]
        n_avg = float(np.mean(nodes))
        e_avg = float(np.mean(edges))
        n     = len(self.nodes())

        m = sum((nodes[i] * edges[i] for i in range(n))) - n * n_avg * e_avg
        m /= sum(n * n for n in nodes) - n * n_avg * n_avg

        logC = e_avg - m * n_avg
        C = exp(logC)

        self._theta, self._C = m, C
        # compute outlierness of each node
        self._outlierness = sorted([(node, self.out_line(node)) for node in self.nodes()], key=lambda t:t[1])

    def e_count(self, node):
        return int(self.egonetworks[node]['edges'])

    def n_count(self, node):
        return int(self.egonetworks[node]['nodes'])

    def nodes(self):
        return self.graph.nodes()

    def out_line(self, node):
        y        = self.egonetworks[node]['edges']
        Cx_theta = self._C * pow(self.n_count(node), self._theta)
        return max(y, Cx_theta)/min(y, Cx_theta)*log(fabs(y-Cx_theta) + 1)
