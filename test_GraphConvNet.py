from unittest import TestCase
import networkx as nx
from unittest.mock import patch, Mock

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


    def test_prob_greedy_alg(self):
        self.assertEqual(self.gcn.greedyAlg(self.gcn.f), set(['a', 'c']))
        self.gcn.g.add_edges_from([('c', 1), ('c', 4)])
        self.assertEqual(self.gcn.greedyAlg(self.gcn.f), set(['c']))

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


    def test_feature_degree(self):
        self.assertEqual(self.gcn.feature_degree()['b'], 1/5)
        self.assertEqual(self.gcn.feature_degree()['a'], 3/5)

    def test_feature_mean_uni_degree(self):
        self.assertEqual(self.gcn.feature_mean_uni_degrees()['b'], 3/3)
        self.assertAlmostEqual(self.gcn.feature_mean_uni_degrees()['a'], ((5/3)/3))

    def test_feature_min_uni_degree(self):
        self.assertEqual(self.gcn.feature_min_uni_degrees()['b'], 3/3)
        self.assertEqual(self.gcn.feature_min_uni_degrees()['a'], 1/3)

    def test_features_to_tensors(self):
        data, subset_to_row = self.gcn.features_to_tensors()
        self.assertEqual(data[subset_to_row['a'], 0], 3/5)
        self.assertAlmostEqual(data[subset_to_row['a'], 1], (5/3)/3)
        self.assertEqual(data[subset_to_row['a'], 2], 1/3)

    def mock_compute_marginal_gain(self, i):
        return {'a': 1, 'b': 2, 'c': 3}

    def test_labels_to_tensors(self):
        data, subset_to_row = self.gcn.features_to_tensors()
        self.gcn.compute_marginal_gain = self.mock_compute_marginal_gain
        labels = self.gcn.compute_marginal_gain(1)
        tensor_labels = self.gcn.labels_to_tensors(labels, subset_to_row)
        self.assertEqual(tensor_labels[subset_to_row['a']], 1)
        self.assertEqual(tensor_labels[subset_to_row['b']], 2)
        self.assertEqual(tensor_labels[subset_to_row['c']], 3)


