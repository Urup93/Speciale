import torch


class GraphNeuralNet:
    def __init__(self):
        self.model = None

    def train(self, features, labels, epochs=100):
        number_of_nodes, number_of_features = features.size()
        hidden_nodes_layer_one = 10
        hidden_nodes_layer_two = 1
        if self.model is not None:
            model = self.model
        else:
            model = torch.nn.Sequential(
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(number_of_features, hidden_nodes_layer_one),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(hidden_nodes_layer_one, hidden_nodes_layer_two),
                torch.nn.ReLU()
            )
        loss_fn = torch.nn.MSELoss(reduction='mean')
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for i in range(epochs):
            predictions = model(features)
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.model = model

    def predict(self, features):
        return self.model(features)

    def score(self, features, labels):
        pred = self.predict(features)
        score = 1/len(pred)*(sum((labels-pred)**2))
        return score