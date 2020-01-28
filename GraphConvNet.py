import torch
import numpy as np
import networkx as nx

class GraphConvNet:

    def __init__(self, g):
        self.g = g

    # Optimazation function
    def f(self, V, v):
        nb = [u for u in self.g.neighbors(v) if u in V]
        return len(nb)


    # f optimazation function
    def greedyAlg(self, func):
        solution = set()
        elements_covered = set()
        universe = set([u for u,d in self.g.nodes(data=True) if d["bipartite"] == 1])
        subsets = set([u for u, d in self.g.nodes(data=True) if d["bipartite"] == 0])
        while elements_covered != universe:
            elements_not_covered = [u for u in universe if u not in elements_covered]
            available_subsets = [s for s in subsets if s not in solution]
            scores_for_subsets = [func(elements_not_covered, s) for s in available_subsets]
            idx = np.argmax(scores_for_subsets)
            best_subset = available_subsets[idx]
            [elements_covered.add(u) for u in self.g.neighbors(best_subset)]
            solution.add(best_subset)
        return solution


    def GCN(self, score, features, depth, Weight_mat, weight_vec):
        h_old = features
        for k in range(depth):
            for v in self.g.nodes:
                neighbours = self.g.neightbors(v)
                print(neighbours)
                h = np.mean(np.dot(weight_vec, h_old[neighbours].T), axis=0)
                print(h)
        return self.g
