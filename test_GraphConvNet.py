from unittest import TestCase
import networkx as nx

from GraphConvNet import GraphConvNet


class TestGraphConvNet(TestCase):
    def setUp(self):
        G = nx.Graph()
        G.add_nodes_from(["a","b","c"], bipartite = 0)
        G.add_nodes_from([2, 1, 4, 0, 3], bipartite = 1)
        G.add_edges_from([("a", 2), ("a", 1), ("a", 4)])
        G.add_edges_from([("b", 2)])
        G.add_edges_from([("c", 2), ("c", 0), ("c", 3)])
        self.gcn = GraphConvNet(G)

    def test_score_sum_of_neighbors(self):
        self.assertEqual(self.gcn.f([0, 1, 2, 3, 4], "a"), 3)
        self.assertEqual(self.gcn.f([0, 2, 3, 4], "a"), 2)
        self.assertEqual(self.gcn.f([0, 3, 4], "b"), 0)
        self.assertEqual(self.gcn.f([0, 1, 4], "c"), 1)
        pass

    def test_greedy_alg(self):
        func = self.gcn.f
        self.assertEqual(self.gcn.greedyAlg(func),set(["a", "c"]))
        self.gcn.g.add_edges_from([("c", 4), ("c", 1)])
        self.assertEqual(self.gcn.greedyAlg(func), set(["c"]))
        pass

    def test_gcn(self):
        pass
