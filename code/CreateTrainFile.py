#Georgios Vletsas 4924
#Stefanos Gersch-Koutsogiannis 5046

import pandas as pd
import networkx as nx
import random

# Διαβαζω τον πλήρη γράφο 
edge_df = pd.read_csv("edgelist.txt", header=None, sep=",", names=["source", "target"])
G_full = nx.from_pandas_edgelist(edge_df, source="source", target="target")

# Διαβαζω τις ακμές του test set
test = pd.read_csv("test.txt", header=None, names=["source", "target"])
test_edges = set([tuple(x) for x in test.values])


#κρατάω όλους τους κόμβους, αφαιρώ μόνο τις ακμές του test set
G_train = nx.DiGraph()
G_train.add_nodes_from(G_full.nodes())
train_edges = [e for e in G_full.edges() if e not in test_edges]
G_train.add_edges_from(train_edges)

# Δημιουργία θετικών δειγμάτων
positive_samples = [(u, v, 1) for u, v in G_train.edges()]

# Δημιουργία αρνητικών δειγμάτων
nodes = list(G_train.nodes())
existing_edges = set(G_train.edges()) | set((v, u) for u, v in G_train.edges())#emeine otan animetopizame to grafo san mh kateyueinomeno grafo
num_neg = len(positive_samples)
negative_samples = set()

while len(negative_samples) < num_neg:
    u, v = random.sample(nodes, 2)
    if (u, v) not in existing_edges and (v, u) not in existing_edges:#emeine otan animetopizame to grafo san mh kateyueinomeno grafo
        negative_samples.add((u, v))

negative_samples = [(u, v, 0) for u, v in negative_samples]

# Συνδυασμός, ανακάτεμα και αποθήκευση
train_data = positive_samples + negative_samples
random.shuffle(train_data)

train = pd.DataFrame(train_data, columns=["source", "target", "label"])
train.to_csv("train.txt", index=False, header=False)

print("train.txt created")
