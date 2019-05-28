#!/usr/bin/env python3
import sys

# Sanity check on the python environment
try:
    import argparse
    import collections
    import itertools
    import matplotlib.pyplot
    import networkx
    import numpy
    import powerlaw
    import random
    import scipy
    import scipy.special
    import scipy.stats
    # from type import List
except ImportError as ie:
    print(f'Missing libraries: {ie}')


class GraphAnalyser(networkx.Graph):
    """
    Graph analyser class.
    It integrates the graph and directed graphs classes from networkx
    """

    def __init__(
            self,
            filename: str,
            delimiter: str = '\t',
            directed: bool = False
    ):
        """
        Initialise the graph as undirected/directed
        :param directed: boolean saying whether the graph is directed or not
        """
        networkx.Graph.__init__(self)
        if directed:
            self.to_directed()

        with open(filename) as f:
            edgelist = list(
                map(lambda s: s.strip().split(delimiter), f.readlines())
            )
        self.add_edges_from(edgelist)
        self.pmf = None
        self.cdf = None
        self.fit = None
        self.average_degree = 0
        self.connected_components = 0
        self.is_connected = False
        self.diameter = 0

    def __pmf(self, normalised: bool = False):
        """
        Calculate the PMF of the degree distribution
        :param normalised: If true the histogram is normalised
        :return:
        """
        try:
            self.pmf = numpy.array(
                networkx.degree_histogram(self),
                dtype=numpy.float
            )
            if normalised:
                self.pmf /= self.number_of_nodes()
        except Exception as e:
            print(f'Impossible to calculate PDF: {e}')

    def __cdf(self):
        """
        Calculate the CDF of the degree distribution
        :return:
        """
        try:
            self.cdf = numpy.fromiter(
                itertools.accumulate(self.pmf),
                dtype=numpy.float
            )
        except Exception as e:
            print(f'Impossible to calculate CDF: {e}')

    def __avg_deg(self, verbose: bool = False):
        """
        Calculate the average degree
        :param verbose: Print the average degree if required
        :return:
        """
        self.average_degree = sum(networkx.degree_histogram(self)) / \
                              self.number_of_nodes()
        if verbose:
            print(f'Average Degree: {self.average_degree: 6.2f}')

    def __conn(self, verbose: bool = False):
        self.is_connected = networkx.is_connected(self)
        self.connected_components = networkx.number_connected_components(self)
        if verbose:
            print(
                f'Found {self.connected_components: d} connected components.'
            )

    def __dia(self):
        large = next(
            self.subgraph(c)
            for c in networkx.strongly_connected_components(self)
        )
        self.diameter = networkx.diameter(large)

    def analyse(self, normalised: bool = False):
        """
        Analyse the graph and calculate the main statistics
        :return:
        """
        self.__pmf(normalised)
        self.__cdf()
        self.__avg_deg()


    # def plot(self, plot_type: str, attrs, markers):
    #     """
    #     Plot the required
    #     :param plot_type: the type of the plot
    #     :param attrs: the attributes to be shown on the plot
    #     :param markers: the markers to be used for the plots
    #     :return:
    #     """
    #     try:
    #         plotfun = eval(f'matplotlib.pyplot.{plot_type}')
    #     except AttributeError:
    #         print(
    #             f'{plot_type} is not a legal matplotlib function',
    #             file=sys.stderr
    #         )
    #     try:
    #         if isinstance(attrs, str):
    #             var = dict(self.__getattribute__(attrs))
    #             x = var.keys()
    #             y = var.values()
    #             plotfun(x, y, markers)
    #         elif isinstance(attrs, list) || isinstance(attrs, tuple):
    #             if all(map(lambda s: isinstance(s, tuple), attrs)):
    #                 for t in attrs:


def degree_distribution(graph):
    """
    Calculate the actual degree distribution of a network.
    :param graph: the networkx Graph() object
    :return: the degree distribution as a counter {degree: number of nodes}
    """
    d = sorted(graph.degree, key=lambda x: x[1])
    return collections.Counter(map(lambda deg: deg[1], d))


def degree_cumulative_distribution(graph, p=False, com=False):
    """
    Calculate the actual cumulative distribution
    :param graph: the networkx Graph() object
    :param p: if True the cumulative distribution is plotted
    :param com: if True plot the complement to one
    :return:
    """
    d = degree_distribution(graph)
    p_k = numpy.fromiter(d.values(), numpy.float) / sum(d.values())
    k = numpy.fromiter(d.keys(), numpy.float)
    _P_k = numpy.fromiter(itertools.accumulate(p_k), dtype=numpy.float)
    if p:
        if com:
            matplotlib.pyplot.loglog(k, 1 - _P_k, 'k*')
            matplotlib.pyplot.ylabel('1 - P(k)')
        else:
            matplotlib.pyplot.loglog(k, _P_k, 'k*')
            matplotlib.pyplot.ylabel('P(k)')

        matplotlib.pyplot.xlabel('k')
        matplotlib.pyplot.show()
    return _P_k


def degree_pdf(k, k_min):
    """
    Calculate the PDF(k) to be fitted
    :param k: array of nodes degree
    :param k_min:
    :return:
    """
    n = len(k)
    g = 1 + n / sum(scipy.log(k) / scipy.log(k_min - .5))
    pdf = 1 / (scipy.power(k, g) * scipy.special.zeta(g, k_min))
    # print(f'\u03B3: {g: f.4}')
    matplotlib.pyplot.loglog(k, pdf, 'g-')
    matplotlib.pyplot.show()
    return pdf


def degree_cdf(k, k_min):
    """
    Calculate the CDF(k) to be fitted
    :param k: array of nodes degrees
    :param k_min:
    :return:
    """
    n = len(k)
    g = 1 + n / sum(scipy.log(k) / scipy.log(k_min - .5))
    cdf = 1 - scipy.special.zeta(k, g) / scipy.special.zeta(k_min, g)
    print(f'\u03B3: {g: f}')
    return cdf


def fit_powerlaw(graph: networkx.Graph):
    """
    Calculate the value of K_min that minimises the Kolmogorov-Smirnof test
    :param graph: the networkx Graph() object
    :return: a value of K_min
    """
    k = list(dict(graph.degree).values())
    dist = powerlaw.Fit(k)
    return dist


def degree_poisson(graph: networkx.Graph):
    """
    Calculate the poisson distribution of a given graph
    :param graph: the networkx Graph() object
    :return: the poisson distribution
    """
    # TODO investigate how the scipy poisson distribution actually works
    avg_k = average_degree(graph)
    k = numpy.fromiter(degree_distribution(graph).keys(), dtype=numpy.float)
    pdf = scipy.stats.poisson(avg_k)
    pdf.pmf(k)
    matplotlib.pyplot.plot(k, pdf.pmf(k))
    matplotlib.pyplot.show()
    return pdf


def average_degree(graph: networkx.Graph) -> float:
    """
    Calculate the average degree of the network
    :param graph: the networkx Graph() object
    :return: the average degree of the network
    """
    k = map(lambda d: d[1], graph.degree())
    return sum(list(k)) / graph.number_of_nodes()


def plot_pdf(graph: networkx.Graph, log_b=True):
    """
    Plot the degree distribution with relevant power law and poisson
    distribution fits
    :param graph: the networkx Graph() object
    :param log_b: plot the log binned distribution if required
    :return: plot of the distribution
    """
    matplotlib.pyplot.figure()
    matplotlib.pyplot.grid(True)
    deg = graph.degree()
    d = degree_distribution(graph)
    k = numpy.fromiter(d.keys(), dtype=numpy.int)
    p = numpy.fromiter(
        map(lambda x: x / len(deg), d.values()),
        # d.values(),
        dtype=numpy.float
    )
    r = powerlaw.Fit(
        numpy.fromiter(dict(deg).values(), dtype=numpy.float) / len(deg)
    )
    # p = p / len(deg)
    matplotlib.pyplot.loglog(k, p, 'ro', alpha=.3)
    matplotlib.pyplot.ylabel('p(k)')
    matplotlib.pyplot.xlabel('k')
    if log_b:
        binned = log_binning(graph)
        binned_k = [i['avg_k'] for i in binned]
        binned_p = [i['p_k'] for i in binned]
        matplotlib.pyplot.loglog(binned_k, binned_p, 'b+')

    # pdf = scipy.power(k, -r.alpha) * scipy.power(r.xmin, r.alpha - 1) * (
    #         r.alpha - 1
    # )
    pdf = scipy.power(k, -r.alpha) / scipy.special.zeta(r.xmin, r.alpha)
    matplotlib.pyplot.loglog(k, pdf, 'm')
    # r.plot_pdf()
    matplotlib.pyplot.show()


def graph_from_edgelist(path_to_file: str, separator='\t') -> networkx.Graph:
    """
    Return a networkx graph object from an edgelist
    :param path_to_file: absolute path to the edge list
    :param separator: the separator between nodes of an edge
    :return:  Graph object
    """
    return networkx.read_edgelist(path_to_file, delimiter=separator)


def log_binning(graph: networkx.Graph, base: int = 2) -> list:
    """
    Bin the distribution according to a logarithmic law
    :param base: base of the logarithmic distribution
    :param graph: the Graph() object to study
    :return bins: the dictionary of the bins
    """
    if base == 1:
        raise ValueError

    n = graph.number_of_nodes()
    dist = degree_distribution(graph)
    k_max = max(list(dist.keys()))
    number_of_bins = int(scipy.logn(base, k_max)) + 1
    print(f'Dividing the degree distribution in {number_of_bins: d} bins')

    bins = [
        {'n': 0, 'n*k': 0, 'avg_k': 0.0, 'p_k': 0.0}
        for _ in range(number_of_bins)
    ]
    for k in dist.keys():
        bins[int(scipy.logn(base, k))]['n'] += dist[k]
        bins[int(scipy.logn(base, k))]['n*k'] += dist[k] * k

    for i, _bin in enumerate(bins):
        _bin['avg_k'] = _bin['n*k'] / _bin['n']
        _bin['p_k'] = _bin['n'] / n / scipy.power(base, i)
        # _bin['p_k'] = _bin['n'] / scipy.power(base, i)

    return bins


def random_failures(graph: networkx.Graph):
    """
    simulate random failures in the network
    :param graph: networkx.Graph()
    :return:
    """
    while networkx.is_connected(graph):
        rand_node = random.choice(graph.nodes)
        print(f'Removing node: {rand_node}\n')
        graph.remove_node(rand_node)
    else:
        print('The graph has failed!\n')
        return failed(graph)


def density(graph: networkx.Graph):
    """
    Returns the density of a graph as L / N * (N - 1)
    :param graph: the graph which density is being calculated
    :return: The density of the graph
    """
    l: int = graph.number_of_edges()
    n: int = graph.number_of_nodes()
    return 2 * l / (n * (n - 1))


def failed(graph: networkx.Graph):
    """
    Return the the connected components if the network failed; else it will
    return false.
    :param graph: The graph to analyse
    :return:
    """
    if networkx.is_connected(graph):
        return False
    else:
        return (
            graph.subgraph(comp)
            for comp in networkx.connected_components(graph)
        )


def correlation(graph: networkx.Graph):
    return networkx.correlation.degree_mixing_matrix(graph)


def assortativity(graph: networkx.Graph):
    return networkx.correlation.degree_assortativity_coefficient(graph)


if __name__ == 'main':
    ga = argparse.ArgumentParser()
    ga.add_argument(
        "-d",
        "--degree-distribution",
        help="plot the degree distribution on a log-log scale",
        type=bool
    )
    ga.add_argument(
        "-k",
        "--average-degree",
        help="print the average degree",
        type=bool
    )
    ga.add_argument(
        "-c",
        "--average-clustering-coefficient",
        help="print the average clustering coefficient",
        type=bool
    )
    ga.parse_args()
