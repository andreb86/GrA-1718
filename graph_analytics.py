#!/usr/bin/env python3
import sys

# Sanity check on the python environment
try:
    import argparse
    import collections
    import matplotlib.pyplot
    import networkx
    import numpy
    #import powerlaw
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

    def __pmf(self, normalise: bool = True):
        """
        Calculate the PMF of the degree distribution
        :param normalise: If true the histogram is normalised
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
        self.betweenness = sorted(
            networkx.betweenness_centrality(
                self,
                normalized=True
            ),
            key=lambda x: x[1],
            reverse=True
        )

    def component(self, idx: int = 0, in_place: bool = False):
        """
        Return the component referenced by the index
        :param idx: index of the component the graph in the component list.
        Default to the largest component.
        :param in_place: if True the graph is shrunk to the desired component
        :return:
        """
        if self.connected_components == 0:
            print("Please analyse graph first.", file=sys.stderr)
            raise ValueError
        if idx > self.connected_components:
            print("The requested component does not exist", file=sys.stderr)
            raise IndexError
        cl = networkx.strongly_connected_components(self)
        count = 0
        for nodes_set in cl:
            if in_place:
                if count != idx:
                    self.remove_nodes_from(nodes_set)
            else:
                if count == idx:
                    sub_graph = networkx.subgraph(self, nodes_set)
            count += 1
        if in_place:
            return self
        else:
            return sub_graph

    def analyse(self, normalised: bool = True):
        """
        Analyse the graph and calculate the main statistics
        :return:
        """
        self.__pmf(normalised)
        self.__cdf()
        self.__avg_deg()
        self.__conn()
        self.__bet()

    def attack(self, mode: str = 'random'):
        """
        Attack the network by deleting the nodes
        :param mode: The attack mode. Possible values are 'random' or
        'betweenness'
        :return: None
        """
        # Fetch the biggest connected component
        cc = self.component(0)

        if mode == 'random':
            numpy.random.seed(time())
            while networkx.is_connected(cc):
                cc.remove_node(numpy.random.choice(cc.nodes))
            else:
                print("The largest component of the graph has failed!")
        elif mode == 'betweenness':
            for node in self.betweenness:
                print(f"Removing node: {node: s}")
                if networkx.is_connected():
                    cc.remove_node(node)
                else:
                    print(
                        "The largest connected component has failed.",
                        file=sys.stderr
                    )
        else:
            raise ValueError("Unexpected attack mode!")

    # TODO encapsulate the method into a fit and make it "private"
    def k_min(self):
        """
        Find the Kmin parameter of the distribution
        :return: the Kmin parameter
        """
        dist = []
        for (k, h) in enumerate(self.pmf):
            if k:
                gamma = 1 + \
                        self.number_of_nodes() / \
                        numpy.sum(numpy.log(self.pmf / (k - 1 / 2)))
                # Calculate the auxiliary variable for the KS test
                x = 1 / (zeta(gamma, k) * self.degree)
                d = kstest(x, 'powerlaw', gamma)
                dist.append(d.statistic)
