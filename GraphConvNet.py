import torch
import numpy as np
import networkx as nx
import random

class GraphConvNet:

    def __init__(self, g):
        self.g = g
        self.subsets, self.universe = nx.algorithms.bipartite.sets(g)

    # Optimazation function
    def f(self, V, v):
        """
        Scores a subset based on how many uncovered neighbors it has.
        :param V: List of all uncovered elements
        :param v: Subset to inspect/score
        :return: score of v
        """
        nb = [u for u in self.g.neighbors(v) if u in V]
        return len(nb)

    def greedyAlg(self, func):
        """
        Find solutions based on the greedy approach based on the optimazation function func
        :param func: function to score each subset.
        :return: solution
        """
        solution = set()
        elements_covered = set()
        while elements_covered != self.universe:
            elements_not_covered = [u for u in self.universe if u not in elements_covered]
            available_subsets = [s for s in self.subsets if s not in solution]
            scores_for_subsets = [func(elements_not_covered, s) for s in available_subsets]
            idx = np.argmax(scores_for_subsets)
            best_subset = available_subsets[idx]
            [elements_covered.add(u) for u in self.g.neighbors(best_subset)]
            solution.add(best_subset)
        return solution

    def probalisticGreedyAlg(self, func):
        """
        Find solutions based on the greedy approach based on the optimazation function func
        :param func: function to score each subset.
        :return: solution
        """
        solution = set()
        elements_covered = set()
        (subsets, universe) = nx.algorithms.bipartite.sets(self.g)
        gain = {s: 0 for s in subsets}
        while elements_covered != universe:
            elements_not_covered = [u for u in universe if u not in elements_covered]
            available_subsets = [s for s in subsets if s not in solution]
            scores_for_subsets = {s: func(elements_not_covered,s) for s in available_subsets}
            sum_of_scores = sum(scores_for_subsets.values())
            probs = [scores_for_subsets[s]/sum_of_scores for s in scores_for_subsets.keys()]
            chosen_subset = random.choices(list(scores_for_subsets.keys()), weights=probs)[0]
            gain[chosen_subset] = scores_for_subsets[chosen_subset]
            solution.add(chosen_subset)
            [elements_covered.add(u) for u in self.g.neighbors(chosen_subset)]
        return solution, gain

    def compute_marginal_gain(self, iterations):
        total_gain = {s: 0 for s in self.subsets}
        for i in range(iterations):
            gain = self.probalisticGreedyAlg(self.f)[1]
            total_gain = {s: gain[s] + total_gain[s] for s in self.subsets}
        marg_gain = {s: total_gain[s] / iterations for s in self.subsets}
        return marg_gain


    def feature_degree(self):
        """
        Computes the proportion of elements in the universe that each subset covers, and outputs
        a dictionary of all subsets corrosponding proportions.
        :return: dictionary that maps a subset to its proportion of elements of covered in the graph.
        """
        number_of_elements = len(self.universe)
        std_degrees = {s: self.g.degree[s]/number_of_elements for s in self.subsets}
        return std_degrees


    def feature_mean_uni_degrees(self):
        """
        For each subset in the graph, goes through all the elements it is covering and for each element
        computes the proportion of subsets that covers it. Then takes the mean over all elements covered
        by this subset, and outputs this as a feature for the subset.
        :return: A dictionary mapping all subsets to the mean degree of its elements
        """
        mean_uni_degrees = {}
        number_of_subsets = len(self.subsets)
        for s in self.subsets:
            sum = 0
            for n in self.g[s]:
                sum += self.g.degree[n]/number_of_subsets
            mean = sum/self.g.degree[s]
            mean_uni_degrees[s] = mean
        return mean_uni_degrees


    def feature_min_uni_degrees(self):
        """
        For each subset in the graph, find the element it covers with the lowest degree and output the
        proportion of subset this element is covered by.
        :return: dictionary mapping all subsets to the degree of its lowest connected neighbor.
        """
        min_uni_degree = {}
        number_of_subsets = len(self.subsets)
        for s in self.subsets:
            min_degree = float('inf')
            for n in self.g[s]:
                cur = self.g.degree[n]
                if min_degree > cur:
                    min_degree = cur
            min_uni_degree[s] = min_degree/number_of_subsets
        return min_uni_degree


    def features_to_tensors(self):
        """
        Converts features of a graph into a tensor matrix
        :return: Pytorch tensor matrix
        """
        degree_dict = self.feature_degree()
        mean_dict = self.feature_mean_uni_degrees()
        min_dict = self.feature_min_uni_degrees()
        data = torch.empty(len(self.subsets), 3)
        i = 0
        subset_to_row = {}
        for s in self.subsets:
            data[i, 0] = degree_dict[s]
            data[i, 1] = mean_dict[s]
            data[i, 2] = min_dict[s]
            subset_to_row[s] = i
            i += 1
        return data, subset_to_row


    def labels_to_tensors(self, labels_dict, subset_to_row):
        labels = torch.empty(len(self.subsets),1)
        for l in labels_dict:
            idx = subset_to_row[l]
            labels[idx] = labels_dict[l]
        return labels

    def GCN(self, labels, features, depth, Weight_mat, weight_vec):

        return 0
