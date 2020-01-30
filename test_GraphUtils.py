from unittest import TestCase
import networkx as nx
from unittest.mock import patch, Mock
import GraphUtils as gu


class TestGraphUtils(TestCase):
    def setUp(self):
        graph = nx.Graph()
        graph.add_nodes_from(["a", "b", "c"], bipartite=0)
        graph.add_nodes_from([2, 1, 4, 0, 3], bipartite=1)
        graph.add_edges_from([("a", 2), ("a", 1), ("a", 4)])
        graph.add_edges_from([("b", 2)])
        graph.add_edges_from([("c", 2), ("c", 0), ("c", 3)])
        self.graph = graph

    def test_score_sum_of_neighbors(self):
        self.assertEqual(gu.f(self.graph, [0, 1, 2, 3, 4], "a"), 3)
        self.assertEqual(gu.f(self.graph, [0, 2, 3, 4], "a"), 2)
        self.assertEqual(gu.f(self.graph, [0, 3, 4], "b"), 0)
        self.assertEqual(gu.f(self.graph, [0, 1, 4], "c"), 1)


    def test_prob_greedy_alg(self):
        self.assertEqual(gu.greedyAlg(self.graph, gu.f), set(['a', 'c']))
        self.graph.add_edges_from([('c', 1), ('c', 4)])
        self.assertEqual(gu.greedyAlg(self.graph, gu.f), set(['c']))

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
        self.assertEqual(gu.probalisticGreedyAlg(self.graph, gu.f)[0], set(['a', 'c']))
        # Tests gain when picking a then c
        self.assertEqual(gu.probalisticGreedyAlg(self.graph, gu.f)[1]["a"], 3)
        self.assertEqual(gu.probalisticGreedyAlg(self.graph, gu.f)[1]["c"], 2)
        # Tests stopping criterion
        self.graph.add_edges_from([('a', 0), ('a', 3)])
        self.assertEqual(gu.probalisticGreedyAlg(self.graph, gu.f)[0], set(['a']))
        self.assertEqual(gu.probalisticGreedyAlg(self.graph, gu.f)[1]["a"], 5)


    def test_feature_degree(self):
        self.assertEqual(gu.feature_degree(self.graph)['b'], 1/5)
        self.assertEqual(gu.feature_degree(self.graph)['a'], 3/5)

    def test_feature_mean_uni_degree(self):
        self.assertEqual(gu.feature_mean_uni_degrees(self.graph)['b'], 3/3)
        self.assertAlmostEqual(gu.feature_mean_uni_degrees(self.graph)['a'], ((5/3)/3))

    def test_feature_min_uni_degree(self):
        self.assertEqual(gu.feature_min_uni_degrees(self.graph)['b'], 3/3)
        self.assertEqual(gu.feature_min_uni_degrees(self.graph)['a'], 1/3)

    def test_features_to_tensors(self):
        data, subset_to_row = gu.features_to_tensors(self.graph)
        self.assertEqual(data[subset_to_row['a'], 0], 3/5)
        self.assertAlmostEqual(data[subset_to_row['a'], 1], (5/3)/3)
        self.assertEqual(data[subset_to_row['a'], 2], 1/3)

    def mock_compute_marginal_gain(self, graph, i):
        return {'a': 1, 'b': 2, 'c': 3}

    def test_labels_to_tensors(self):
        data, subset_to_row = gu.features_to_tensors(self.graph)
        gu.compute_marginal_gain = self.mock_compute_marginal_gain
        labels = gu.compute_marginal_gain(self.graph, 1)
        tensor_labels = gu.labels_to_tensors(self.graph, labels, subset_to_row)
        self.assertEqual(tensor_labels[subset_to_row['a']], 1)
        self.assertEqual(tensor_labels[subset_to_row['b']], 2)
        self.assertEqual(tensor_labels[subset_to_row['c']], 3)

    #def test_gcn_train(self):
    #    features, subset_to_row = self.gcn.features_to_tensors()
    #    labels_dict = self.gcn.compute_marginal_gain()
    #    labels = self.gcn.labels_to_tensors(labels_dict, subset_to_row)
    #    self.gcn.gcn_train(features, labels, epochs=10)
