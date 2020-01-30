import torch
from GraphNeuralNet import GraphNeuralNet
import GraphUtils as gu
import GreedySelection as gs


def generate_or_load(generate_tensors):
    if generate_tensors:
        graphs = gu.generate_data(3, (50, 100), (200, 300))
        data, labels = gu.graphs_to_tensor(graphs)
        print('Saving torch data')
        torch.save(data, 'data/data.pt')
        torch.save(labels, 'data/labels.pt')
    else:
        data = torch.load('data/data.pt')
        labels = torch.load('data/labels.pt')
    return data, labels

generate_tensors = False
data, labels = generate_or_load(generate_tensors)

model = GraphNeuralNet()
model.train(data, labels)

g = gu.generate_data(1, (10, 20), (50, 100))
test_data, test_labels = gu.graphs_to_tensor(g)
print(model.score(test_data, test_labels))
solution = gs.greedy_selection(g[0], model)
gu.plot_graph(g[0])