from unittest import TestCase
import networkx as nx

from GraphConvNet import GraphConvNet


class TestGraphConvNet(TestCase):
    def setUp(self):
        G = nx.Graph()
        G.add_nodes_from([2, 1, 4, 0, 3])
        G.add_edges_from([(0, 2), (1, 0), (3, 2)])
        self.gcn = GraphConvNet(G)

    def test_score_sum_of_neighbors(self):
        self.assertEqual(self.gcn.f([0, 1, 2, 3, 4], 0), 2)
        self.assertEqual(self.gcn.f([0, 2, 3, 4], 0), 1)
        self.assertEqual(self.gcn.f([0, 3, 4], 0), 0)
        self.assertEqual(self.gcn.f([0, 1, 3, 4], 4), 0)
        pass

    def test_greedy_alg(self):
        pass

    def test_gcn(self):
        pass
