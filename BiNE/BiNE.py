import numpy as np
import random
import time
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import pandas as pd

def compute_two_hop_adj(adj):
    return np.dot(adj.T,adj)


def random_walk(adj, start, length, alpha = 1):
    path = [start]
    random_neighbor = start
    i = 0
    #print("Starting at: ", start)
    while i/2 < length:
        if i%2 == 0:
            neighbours = np.where(adj[random_neighbor, :] == 1)[0]
            #print(random_neighbor, " covers the elements: ", neighbours)
            random_neighbor = np.random.choice(neighbours)
            #print("Chose: ", random_neighbor)
        elif alpha > random.random():
            neighbours = np.where(adj[:, random_neighbor] == 1)[0]
            #print(random_neighbor, " is covered by ", neighbours)
            random_neighbor = np.random.choice(neighbours)
            #print("Choose: ", random_neighbor)
            path.append(random_neighbor)
        else:
            random_neighbor = start
            path.append(random_neighbor)
        i += 1
    return [str(node) for node in path]


def generate_walks(adj, walk_per_node = 10, walk_length = 40, alpha = 1):
    walks = []
    for start in adj[:,0]:
        for walk in range(walk_per_node):
            walks.append(random_walk(adj, start, walk_length, alpha))
    return walks


def plot_similarity(model):
    nodes = list(model.wv.vocab)
    X = model[nodes]

    pca = PCA(n_components=2)
    pca_x = pca.fit_transform(X)

    plt.scatter(pca_x[:, 0], pca_x[:, 1])

    for i, word in enumerate(model.wv.vocab):
        plt.annotate(word, xy=(pca_x[i, 0], pca_x[i, 1]))

    plt.show()



adj = np.array([[0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 1],
                [0, 1, 1, 1]])


print("Create random walks")
t = time.time()
walks = generate_walks(adj, 40, 5, 0.9)
print("Generated {} walks in {} seconds".format(len(walks), time.time() - t))

print("Training the model")
t = time.time()
model = Word2Vec(walks, size = 2, window = 10, min_count = 0, sg = 1, hs = 1, workers = 1, iter = 10)
print("Training time:", time.time() - t)

plot_similarity(model)

for i in range(np.shape(adj)[0]):
    print("Similarity for node ", i, ": ", model.most_similar(str(i)))
