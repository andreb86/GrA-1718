#!/usr/bin/env python3
import sys

# Sanity check on the python environment
try:
    import argparse
    import collections
    import matplotlib.pyplot
    import networkx
    import numpy
    # import powerlaw
    import random
    import scipy
    from itertools import accumulate
    from scipy.special import zeta
    from scipy.stats import powerlaw
    from scipy.stats import poisson
    from scipy.stats import kstest
    from time import time
    # from type import List
except ImportError as ie:
    print(f'Missing libraries: {ie}')
except:
    print("Something went wrong!!!")


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
        :param filename: the path to the edgelist
        :param delimiter: the node separator
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
        self.binned_pmf = None
        self.cdf = None
        self.fit = None
        self.average_degree = 0
        self.connected_components = 0
        self.is_connected = False
        self.diameter = 0
        self.betweenness = None
        self.hubs = None
        self.authorities = None
        self.diameter = 0

    def __pmf(self, normalise: bool = True):
        """
        Calculate the PMF of the degree distribution
        :param normalise: If true the histogram is normalised to unity
        :return:
        """
        try:
            self.pmf = numpy.array(
                networkx.degree_histogram(self),
                dtype=numpy.float
            )
            if normalise:
                self.pmf /= self.pmf.sum()
            self.binned_pmf = scipy
        except (TypeError, ZeroDivisionError) as e:
            print(f'Impossible to calculate PMF: {e}')

    def __cdf(self):
        """
        Calculate the CDF of the degree distribution
        :return:
        """
        try:
            self.cdf = numpy.fromiter(accumulate(self.pmf), dtype=numpy.float)
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
        """
        Determine if the network is connected
        :param verbose: if True prints the number of connected components
        :return: None
        """
        self.is_connected = networkx.is_connected(self)
        self.connected_components = networkx.number_connected_components(self)
        if verbose:
            print(
                f'Found {self.connected_components: d} connected components.'
            )

    def __bet(self):
        """
        Calculate the betweenness of the graph
        :return:
        """
        self.betweenness = dict(sorted(
            networkx.betweenness_centrality(
                self,
                normalized=True
            ),
            key=lambda x: x[1],
            reverse=True
        ))

    def __hits(self):
        """
        Determine the hub and authority of each node in the graph and sort
        them based on their HITS score (hubs and authorities)
        :return:
        """
        tmp = networkx.hits(self)
        self.hubs = dict(sorted(
            tmp[0].items(), key=lambda x: x[1],
            reverse=True
        ))
        self.authorities = dict(sorted(
            tmp[1].items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def __clu(self):
        """
        Determine the clustering coefficient of every node of the graph and
        sort them accordingly
        :return:
        """
        self.clustering = dict(sorted(
            networkx.clustering(self).items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def __clo(self):
        """
        Determine the closeness of each node and sort them accordingly
        :return:
        """
        self.closeness = dict(sorted(
            networkx.closeness_centrality(self).items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def component(self, idx: int = 0, in_place: bool = True):
        """
        Return the component referenced by the index
        :param idx: index of the component the graph in the component list.
        :param in_place: True if the subgraph operation must happen in place
        Default to the largest component.
        :return:
        """
        self.__conn()
        if self.connected_components == 0:
            print("Please analyse graph first.", file=sys.stderr)
            raise ValueError
        if idx > self.connected_components:
            print("The requested component does not exist", file=sys.stderr)
            raise IndexError
        components_list = list(networkx.connected_components(self))
        if in_place:
            for i, nset in enumerate(components_list):
                if i != idx:
                    self.remove_nodes_from(nset)
            self.analyse()
        else:
            sub: networkx.Graph = networkx.Graph(self)
            return sub.subgraph(components_list[idx])

    def analyse(self, normalised: bool = True):
        """
        Analyse the graph and calculate the main statistics
        :return:
        """
        self.__pmf(normalised)
        self.__cdf()
        self.__avg_deg()
        self.__conn()
        self.__hits()
        self.__clu()
        # self.__clo()
        # self.diameter = networkx.diameter(self)
        # self.__bet()

    def attack(self, mode: str = 'random'):
        """
        Attack the network by deleting the nodes
        :param mode: The attack mode. Possible values are 'random',
        'betweenness', 'closeness', 'hubs' or 'clustering'
        :return: None
        """

        diam = []
        sizes = []
        avg_k = []
        f = numpy.linspace(0, 1, self.number_of_nodes())
        nodes = None
        if mode == 'random':
            nodes = numpy.asarray(self.nodes)
            numpy.random.seed(time())
            numpy.random.shuffle(nodes)
        elif mode == 'betweenness':
            nodes = numpy.asarray(self.betweenness.keys())
        elif mode == 'closeness':
            nodes = numpy.asarray(self.closeness.keys())
        elif mode == 'hubs':
            nodes = numpy.asarray(self.hubs.keys())
        elif mode == 'clustering':
            nodes = numpy.asarray(self.clustering.keys())
        else:
            raise ValueError("Unexpected attack mode!")

        # TODO
        # Remove every node in the order specified by the attack mode
        for node in nodes:
            try:
                self.remove_node(node)
            except networkx.NetworkXError:
                continue

    def power_fit(self):
        """
        Find approximate gamma and kmin values for the distribution
        :return:
        """
        # k = numpy.arange(1, len(self.pmf))
        # N = self.number_of_nodes()
        # ln_k = numpy.log(k)
        # ln_sum =
        # deg = numpy.fromiter(dict(self.degree).values(), dtype=numpy.float)
        # fit = powerlaw.Fit(deg)
        # kmin = fit.xmin
        # gamma = 1 + N / ()
