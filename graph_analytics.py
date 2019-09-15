#!/usr/bin/env python3
import sys

# Sanity check on the python environment
try:
    import argparse
    import collections
    from copy import deepcopy
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

    def __init__(self, *args, **kwargs, ):
        """
        Initialise the graph as undirected/directed
        :param args: If empty will generate an empty graph, otherwise it will
        accept either an edgelist or another newtworkx.Graph or GraphAnalyser
        (copy constructor)
        :param kwargs: if an edgelist is provided use the 'delimiter' keyword to
        provide the
        :param directed: boolean saying whether the graph is directed or not
        """
        networkx.Graph.__init__(self)
        if not args:
            pass
        elif len(args) > 1:
            print(
                'The number of arguments provided is wrong!',
                file=sys.stderr
            )
            sys.exit(-1)
        else:
            try:
                filename: str = args[0]
                try:
                    delimiter: str = kwargs['delimiter']
                except KeyError:
                    delimiter: str = '\t'
                with open(filename) as f:
                    edgelist = list(
                        map(lambda s: s.strip().split(delimiter), f.readlines())
                    )
                self.add_edges_from(edgelist)
            except (TypeError, FileNotFoundError):
                origin = args[0]
                self.add_edges_from(origin.edges)

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

    def __ave(self, verbose: bool = False):
        """
        Calculate the average degree
        :param verbose: Print the average degree if required
        :return:
        """
        hist = networkx.degree_histogram(self)
        k = numpy.arange(len(hist))
        self.average_degree = numpy.dot(k, hist) / self.number_of_nodes()
        if verbose:
            print(f'Average Degree: {self.average_degree: 6.2f}')
        return self.average_degree

    def __bet(self):
        """
        Calculate the betweenness of the graph
        :return:
        """
        self.betweenness = dict(sorted(
            networkx.betweenness_centrality(self, normalized=True).items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def __cdf(self):
        """
        Calculate the CDF of the degree distribution
        :return:
        """
        try:
            self.cdf = numpy.fromiter(accumulate(self.pmf), dtype=numpy.float)
        except Exception as e:
            print(f'Impossible to calculate CDF: {e}')

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

    def __con(self, verbose: bool = False):
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

    def __dia(self):
        """
        Calculate the diameter of the graph or the largest connected components
        :return:
        """
        self.__con(True)
        if self.is_connected:
            self.diameter = networkx.diameter(self)
        else:
            sub = self.component(in_place=False)
            self.diameter = networkx.diameter(sub)
        return self.diameter

    def __hit(self):
        """
        Determine the hub and authority of each node in the graph and sort
        them based on their HITS score (hubs and authorities)
        :return:
        """
        tmp = networkx.hits(self)
        self.hubs = dict(sorted(
            tmp[0].items(),
            key=lambda x: x[1],
            reverse=True
        ))
        self.authorities = dict(sorted(
            tmp[1].items(),
            key=lambda x: x[1],
            reverse=True
        ))

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

    def component(self,
                  idx: int = 0,
                  in_place: bool = True,
                  analyse: bool = False):
        """
        Return the component referenced by the index
        :param idx: index of the component the graph in the component list.
        :param in_place: True if the sub-graph operation must happen in place
        :param analyse: True if the component must be analysed
        Default to the largest component.
        :return:
        """
        self.__con()
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
            if analyse:
                self.analyse()
        else:
            return self.subgraph(components_list[idx])
            # sub: GraphAnalyser = deepcopy(self)
            # return sub.subgraph(components_list[idx])

    def analyse(self, *args, **kwargs):
        """
        Analyse the graph and calculate the main statistics as requested by
        user. Use 'all' to calculate all of them. The 'exclude' keyword,
        when used in combination with 'all' is used to provide a list of the
        analyses to be excluded. Available analyses are:
        1. average degree
        2. betweenness
        3. CDF (Cumulative Distribution Function)
        4. clustering
        5. closeness
        6. connected
        7. diameter
        8. hits
        9. PMF (Probability Mass Function)
        :param args: analyses to be carried out
        :param kwargs: use the 'exclude' keyword if needed, anything else
        will be discarded.
        :return:
        """
        analyses = {
            'average degree':   self.__ave,
            'betweenness':      self.__bet,
            'CDF':              self.__cdf,
            'clustering':       self.__clu,
            'closeness':        self.__clo,
            'connected':        self.__con,
            'diameter':         self.__dia,
            'hits':             self.__hit,
            'PMF':              self.__pmf
        }

        try:
            mode = args[0]
        except IndexError:
            mode = 'all'

        if mode == 'all':
            try:
                excluded = kwargs['excluded']
                print(f'Excluding from the analysis: {", ".join(excluded)}')
                for key in excluded:
                    try:
                        analyses.pop(key)
                    except KeyError:
                        print(f'The following analysis is unavailable: {key}')
            except KeyError:
                print('Complete analysis selected!\nThis might take a while...')
            if len(args) > 1:
                print('Additional arguments will be ignored...')
            for key in analyses:
                print(f'Executing {key} analysis...')
                analyses[key]()
        else:
            for key in args:
                try:
                    print(f'Executing {key} analysis...')
                    analyses[key]()
                except KeyError:
                    print(
                        f'The following analysis is unavailable: {key}',
                        file=sys.stderr
                    )

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

        print(f'Running attack in {mode} mode\n', file=sys.stderr)
        if mode == 'random':
            nodes = list(self.nodes)
            random.seed(int(time()))
            random.shuffle(nodes)
        elif mode == 'betweenness':
            nodes = list(self.betweenness.keys())
        elif mode == 'closeness':
            nodes = list(self.closeness.keys())
        elif mode == 'hubs':
            nodes = list(self.hubs.keys())
        elif mode == 'clustering':
            nodes = list(self.clustering.keys())
        else:
            raise ValueError("Unexpected attack mode!")

        while len(nodes) > 1:
            try:
                node = nodes[0]
                self.remove_node(node)
                print(f'removing node: {node}')
                self.analyse('average degree')
                avg_k.append(self.average_degree)
                sizes.append(self.number_of_nodes())
                nodes.pop(0)
                print(nodes)
            except networkx.NetworkXError:
                break
        return avg_k, diam, sizes

    def load_attributes(self, filename: str):
        """
        Load the nodes attributes for the contagion
        :param filename: The attributes to be loaded into the nodes
        :return:
        """
        try:
            with open(filename, 'r') as attributes:
                for line in attributes:
                    attr = line.split()
                    node = attr[0]
                    name = attr[1]
                    val = attr[2]
                    self.nodes[node][name] = val
        except FileNotFoundError:
            print('File does not exist', file=sys.stderr)
