import GraphUtils as gu
import torch


def greedy_selection(graph, model):
    solution = []
    while not is_solution(graph):
        data, subset_to_row = gu.features_to_tensors(graph)
        row_to_subset = {v: k for k, v in subset_to_row}
        pred = model.predict(data)
        idx = torch.argmax(pred)
        best_node = row_to_subset[idx]
        graph.remove_nodes_from(graph[best_node])
        graph.remove_node(best_node)
        solution.append(best_node)
        print('yesyes')
    return solution


def is_solution(graph):
    return len(gu.find_universe(graph)) == 0

