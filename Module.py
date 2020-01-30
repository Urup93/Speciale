import networkx as nx
from GraphUtils import GraphUtils
import matplotlib.pyplot as plt

graphs = []
while len(graphs) < 1:
    try:
        cur_graph = nx.algorithms.bipartite.random_graph(6, 6, 0.5)
        gcn = GraphUtils(cur_graph)
        (subsets, universe) = nx.algorithms.bipartite.sets(cur_graph)
        graphs.append(cur_graph)
    except:
        print("Not bipartite graph, retrying")

n = 200
for g in graphs:
    total_gain = {s: 0 for s in subsets}
    for i in range(n):
        solution, gain = gcn.probalisticGreedyAlg(gcn.f)
        total_gain = {s: gain[s]+total_gain[s] for s in subsets}
    marg_gain = {s: total_gain[s]/n for s in subsets}
    print(marg_gain)

print(gcn.compute_marginal_gain(n))



pos = dict()
pos.update((n, (1, i)) for i, n in enumerate(subsets))
pos.update((n, (2, i)) for i, n in enumerate(universe))
nx.draw(g, pos = pos)
plt.show()


#print(gcn.probalisticGreedyAlg(gcn.f))

