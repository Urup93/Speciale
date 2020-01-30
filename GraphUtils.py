import os

import torch
import numpy as np
import networkx as nx
import random

from matplotlib import pyplot as plt


def find_universe(graph):
    universe = set([u for u, d in graph.nodes(data=True) if d["bipartite"] == 1])
    return universe


def find_subsets(graph):
    subsets = set([u for u, d in graph.nodes(data=True) if d["bipartite"] == 0])
    return subsets


def f(graph, uncovered_elements, subset):
    """
    Scores a subset based on how many uncovered neighbors it has.
    :param uncovered_elements: List of all uncovered elements
    :param subset: Subset to inspect/score
    :return: score of subset
    """
    nb = [u for u in graph.neighbors(subset) if u in uncovered_elements]
    return len(nb)


def greedyAlg(graph, func):
    """
    Find solutions based on the greedy approach based on the optimazation function func
    :param graph: Graph
    :param func: Function to score each subset.
    :return: Solution
    """
    subsets = find_subsets(graph)
    universe = find_universe(graph)
    solution = set()
    elements_covered = set()
    while elements_covered != universe:
        elements_not_covered = [u for u in universe if u not in elements_covered]
        available_subsets = [s for s in subsets if s not in solution]
        scores_for_subsets = [func(graph, elements_not_covered, s) for s in available_subsets]
        idx = np.argmax(scores_for_subsets)
        best_subset = available_subsets[idx]
        [elements_covered.add(u) for u in graph.neighbors(best_subset)]
        solution.add(best_subset)
    return solution


def probalisticGreedyAlg(graph, func):
    """
    Find solutions based on the greedy approach based on the optimazation function func
    :param graph: Graph
    :param func: Function to score each subset.
    :return: Solution
    """
    subsets = find_subsets(graph)
    universe = find_universe(graph)
    solution = set()
    elements_covered = set()
    gain = {s: 0 for s in subsets}
    while elements_covered != universe:
        elements_not_covered = [u for u in universe if u not in elements_covered]
        available_subsets = [s for s in subsets if s not in solution]
        scores_for_subsets = {s: func(graph, elements_not_covered, s) for s in available_subsets}
        sum_of_scores = sum(scores_for_subsets.values())
        probs = [scores_for_subsets[s]/sum_of_scores for s in scores_for_subsets.keys()]
        chosen_subset = random.choices(list(scores_for_subsets.keys()), weights=probs)[0]
        gain[chosen_subset] = scores_for_subsets[chosen_subset]
        solution.add(chosen_subset)
        [elements_covered.add(u) for u in graph.neighbors(chosen_subset)]
    return solution, gain


def compute_marginal_gain(graph, iterations=10):
    subsets = find_subsets(graph)
    total_gain = {s: 0 for s in subsets}
    for i in range(iterations):
        gain = probalisticGreedyAlg(graph, f)[1]
        total_gain = {s: gain[s] + total_gain[s] for s in subsets}
    marg_gain = {s: total_gain[s] / iterations for s in subsets}
    return marg_gain


def feature_degree(graph):
    """
    Computes the proportion of elements in the universe that each subset covers, and outputs
    a dictionary of all subsets corrosponding proportions.
    :return: dictionary that maps a subset to its proportion of elements of covered in the graph.
    """
    subsets = find_subsets(graph)
    universe = find_universe(graph)
    number_of_elements = len(universe)
    std_degrees = {s: graph.degree[s]/number_of_elements for s in subsets}
    return std_degrees


def feature_mean_uni_degrees(graph):
    """
    For each subset in the graph, goes through all the elements it is covering and for each element
    computes the proportion of subsets that covers it. Then takes the mean over all elements covered
    by this subset, and outputs this as a feature for the subset.
    :return: A dictionary mapping all subsets to the mean degree of its elements
    """
    subsets = find_subsets(graph)
    mean_uni_degrees = {}
    number_of_subsets = len(subsets)
    for s in subsets:
        sum = 0
        for n in graph[s]:
            sum += graph.degree[n]/number_of_subsets
        mean = sum/graph.degree[s]
        mean_uni_degrees[s] = mean
    return mean_uni_degrees


def feature_min_uni_degrees(graph):
    """
    For each subset in the graph, find the element it covers with the lowest degree and output the
    proportion of subset this element is covered by.
    :return: dictionary mapping all subsets to the degree of its lowest connected neighbor.
    """
    subsets = find_subsets(graph)
    min_uni_degree = {}
    number_of_subsets = len(subsets)
    for s in subsets:
        min_degree = float('inf')
        for n in graph[s]:
            cur = graph.degree[n]
            if min_degree > cur:
                min_degree = cur
        min_uni_degree[s] = min_degree/number_of_subsets
    return min_uni_degree


def features_to_tensors(graph):
    """
    Converts features of a graph into a tensor matrix
    :return: Pytorch tensor matrix and a dictionary mapping subsets to rows in the matrix
    """
    subsets = find_subsets(graph)
    degree_dict = feature_degree(graph)
    mean_dict = feature_mean_uni_degrees(graph)
    min_dict = feature_min_uni_degrees(graph)
    data = torch.empty(len(subsets), 3)
    i = 0
    subset_to_row = {}
    for s in subsets:
        data[i, 0] = degree_dict[s]
        data[i, 1] = mean_dict[s]
        data[i, 2] = min_dict[s]
        subset_to_row[s] = i
        i += 1
    return data, subset_to_row


def labels_to_tensors(graph, labels_dict, subset_to_row):
    subsets = find_subsets(graph)
    labels = torch.empty(len(subsets),1)
    for l in labels_dict:
        idx = subset_to_row[l]
        labels[idx] = labels_dict[l]
    return labels


def graphs_to_tensor(graphs):
    data = None
    labels = None
    for graph in graphs:
        if data is None:
            print('Computing features for initial data')
            data, subset_to_row = features_to_tensors(graph)
            labels_dict = compute_marginal_gain(graph, iterations=20)
            labels = labels_to_tensors(graph, labels_dict, subset_to_row)
            print('Initial data computed')
        else:
            print('Computing features for next graph')
            cur_data, subset_to_row = features_to_tensors(graph)
            labels_dict = compute_marginal_gain(graph, iterations=20)
            cur_labels = labels_to_tensors(graph, labels_dict, subset_to_row)
            print('Concatenating tensor data')
            data = torch.cat((data, cur_data), 0)
            labels = torch.cat((labels, cur_labels), 0)
    return data, labels


def generate_data(amount_of_graphs, range_for_sets, range_for_elements):
    graphs = []
    while len(graphs) < amount_of_graphs:
        sets = random.randrange(range_for_sets[0], range_for_sets[1], step = 1)
        elements = random.randrange(range_for_elements[0], range_for_elements[1], step = 1)
        chance_for_edge = random.uniform(0.1, 0.3) #random.random()
        cur_graph = nx.algorithms.bipartite.random_graph(sets, elements, chance_for_edge)
        has_isolated_node = False
        for v in cur_graph:
            if len(cur_graph[v]) == 0:
                has_isolated_node = True
                break
        if has_isolated_node:
            continue
        graphs.append(cur_graph)
    return graphs


def plot_graph(graph):
    subsets = set([u for u, d in graph.nodes(data=True) if d["bipartite"] == 0])
    universe = set([u for u, d in graph.nodes(data=True) if d["bipartite"] == 1])
    pos = dict()
    pos.update((n, (1, i)) for i, n in enumerate(subsets))
    pos.update((n, (2, i)) for i, n in enumerate(universe))
    nx.draw(graph, pos=pos)
    plt.show()


def store_graphs(graphs):
    i = 1
    name = 'data\graph' + str(i) + '.gefx'
    for g in graphs:
        while os.path.isfile(name):
            i += 1
            name = 'data\graph' + str(i) + '.gefx'
        print('writing ' + name)
        nx.readwrite.write_gexf(g, name)


def load_graph(path):
    if os.path.isfile(path):
        graph = nx.readwrite.read_gexf(path)
    return graph


def load_folder_of_graphs(path):
    graphs = []
    for file in os.listdir(path):
        if file.endswith('.gefx'):
            print('Loading: ', file)
            loc = os.path.join(path, file)
            graphs.append(nx.readwrite.read_gexf(loc))
    return graphs

