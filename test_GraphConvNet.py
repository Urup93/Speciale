from unittest import TestCase
import networkx as nx
from unittest.mock import patch

from GraphConvNet import GraphConvNet


class TestGraphConvNet(TestCase):
    def setUp(self):
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"], bipartite=0)
        G.add_nodes_from([2, 1, 4, 0, 3], bipartite=1)
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

    def test_prob_greedy_alg(self):
        self.assertEqual(self.gcn.greedyAlg(self.gcn.f), set(['a', 'c']))
        self.gcn.g.add_edges_from([('c', 1), ('c', 4)])
        self.assertEqual(self.gcn.greedyAlg(self.gcn.f), set(['c']))
        pass

    def choices(population, weights):
        if "a" in population:
            return "a"
        if "c" in population:
            return "c"
        if "b" in population:
            return "b"

    @patch('random.choices', choices)
    def test_probalistic_greedy_alg(self):
        # Adds a then c to solution
        self.assertEqual(self.gcn.probalisticGreedyAlg(self.gcn.f)[0], set(['a', 'c']))
        # Tests gain when picking a then c
        self.assertEqual(self.gcn.probalisticGreedyAlg(self.gcn.f)[1]["a"], 3)
        self.assertEqual(self.gcn.probalisticGreedyAlg(self.gcn.f)[1]["c"], 2)
        # Tests stopping criterion
        self.gcn.g.add_edges_from([('a', 0), ('a', 3)])
        self.assertEqual(self.gcn.probalisticGreedyAlg(self.gcn.f)[0], set(['a']))
        self.assertEqual(self.gcn.probalisticGreedyAlg(self.gcn.f)[1]["a"], 5)
    pass
