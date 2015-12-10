from math import exp, log, pow

import matplotlib.pyplot as plt
import networkx          as nx


class GraphProp:
    def __init__(self, graph):
        self.graph = graph

        self.m = None
        self.c = None

    def avg_path_length(self):
        if nx.number_connected_components(self.graph) == 1:
            return nx.average_shortest_path_length(self.graph)
        else:
            print('[Error] graph is not connected')

    def degree_distribution(self):
        deg = [self.graph.degree(node) for node in self.graph.nodes()]
        x = sorted(list(set(deg)))
        y = [deg.count(i) for i in x]
        return x, y

    def max_node_degree(self):
        return max(self.graph.degree(n) for n in self.graph.nodes())

    def plot_degree_distribution(self, line=False, axis=None, dot_size=7):
        if line and not (self.m and self.c):
            self.power_law_coefficients()
        x, y = self.degree_distribution()
        # plt.title('Node degree distribution')
        plt.xlabel('Node degree')
        plt.ylabel('Count')
        plt.plot(x, y, 'co', markersize=dot_size)

        if axis:
            plt.axis(axis)
        else:
            plt.axis([0, max(x)*1.2, 0, max(y)*1.2])

        if line:
            yy = [self.c * pow(x[i], self.m) for i in range(len(x))]
            plt.plot(x, yy)

        plt.show()

    def plot_degree_distribution_loglog(self, line=False):
        if line and not (self.m and self.c):
            self.power_law_coefficients()
        x, y = self.degree_distribution()
        plt.title('Node degree distribution logarithmic scale')
        plt.xlabel('Node degree')
        plt.ylabel('Count')
        plt.loglog(x, y, 'co')
        if line:
            yy = [self.c * pow(x[i], self.m) for i in range(len(x))]
            plt.loglog(x, yy)
        plt.show()

    def power_law_coefficients(self):
        x, y = self.degree_distribution()

        n = len(x)
        log_x = [log(i) for i in x]
        log_y = [log(i) for i in y]

        x_avg = sum(log_x) / float(n)
        y_avg = sum(log_y) / float(n)

        m = sum((log_x[i] * log_y[i] for i in range(n))) - n * x_avg * y_avg
        m /= sum(n * n for n in log_x) - n * x_avg * x_avg

        log_c = y_avg * sum(log_x[i] * log_x[i] for i in range(n)) - x_avg * sum(log_x[i] * log_y[i] for i in range(n))
        log_c /= sum(n * n for n in log_x) - n * x_avg * x_avg
        # log_c = y_avg - m * x_avg
        c = exp(log_c)

        self.m, self.c = m, c
        return m, c
