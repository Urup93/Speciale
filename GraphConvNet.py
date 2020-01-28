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
    # b budget (kan udlades hvis set cover og ikke IM)
    def greedyAlg(self, func, b):
        S = list()
        for i in range(b):
            V = [v for v in self.g.nodes if v not in S]
            f_scores = [func(self.g, V, v) for v in V]
            v = np.argmax(f_scores)
            S.append(V[v])
        return S


    def GCN(self, score, features, depth, Weight_mat, weight_vec):
        h_old = features
        for k in range(depth):
            for v in self.g.nodes:
                neighbours = self.g.neightbors(v)
                print(neighbours)
                h = np.mean(np.dot(weight_vec, h_old[neighbours].T), axis=0)
                print(h)
        return self.g
