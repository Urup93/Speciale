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
        print(marg_gain)


    def GCN(self, score, features, depth, Weight_mat, weight_vec):
        return 0