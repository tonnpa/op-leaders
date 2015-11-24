from matplotlib import pyplot as plt

__author__ = 'tonnpa'

class Oddball:

    def __init__(self, graph):
        self._graph       = graph
        self._egonetworks = dict((node, {}) for node in graph.nodes())

        self.run()

    @property
    def egonetworks(self):
        return self._egonetworks

    @property
    def graph(self):
        return self._graph

    def run(self):
        for node in self.nodes():
            # extract egonetwork
            egonetwork = self.graph.subgraph(self.graph.neighbors(node) + [node])
            self._egonetworks[node]['nodes'] = len(egonetwork.nodes())  # number of nodes in egonet
            self._egonetworks[node]['edges'] = len(egonetwork.edges())  # number of edges in egonet

        self.plot()

    def e_count(self, node):
        return int(self.egonetworks[node]['edges'])

    def n_count(self, node):
        return int(self.egonetworks[node]['nodes'])

    def nodes(self):
        return self.graph.nodes()

    def plot(self, threshold=0.75):
        labels, nfeat, efeat = [], [], []
        for node in self.nodes():
            labels.append(node)
            nfeat.append(self.n_count(node))
            efeat.append(self.e_count(node))

        plt.title('Node vs. Edge Feature')
        plt.xlabel('#Nodes')
        plt.ylabel('#Edges')
        x_max = int(max(nfeat)*1.2)
        y_max = int(max(efeat)*1.2)
        plt.axis([0, x_max, 0, y_max])
        # thresholds identifying extremes
        plt.plot([i for i in range(1, x_max)], [i-1 for i in range(1,x_max)])
        plt.plot([i for i in range(1, x_max)], [(i-1)*i/2 for i in range(1, x_max)])
        plt.scatter(nfeat, efeat, c='c')

        for label, x, y in zip(labels, nfeat, efeat):
            if x > x_max*threshold or y > y_max*threshold:
                plt.annotate(
                    label,
                    xy=(x, y), xytext = (-15, 15),
                    textcoords = 'offset points',
                    horizontalalignment = 'right',
                    verticalalignment = 'bottom',
                    arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )

        plt.show()