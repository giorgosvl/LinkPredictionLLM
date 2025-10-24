#Georgios Vletsas 4924
#Stefanos Gersch-Koutsogiannis 5046

import numpy as np
import pandas as pd
import networkx as nx
import os
import pickle
import torch
import torch.nn.functional as F
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from node2vec import Node2Vec
from transformers import AutoTokenizer, AutoModel
from networkx.algorithms.link_prediction import jaccard_coefficient
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


#A: αξιοποιηση γράφου
train = pd.read_csv("train.txt", names=["source", "target", "label"])
train_edges = train[train["label"] == 1]
G_train = nx.from_pandas_edgelist(train_edges, source="source", target="target")

# Δημιουργία Node2Vec μοντέλου
if os.path.exists('node2vec_embeddings.npy') and os.path.exists('node2vec_node_id_map.npy'):
    print("Φόρτωση αποθηκευμένων Node2Vec embeddings")
    embedding_matrix = np.load('node2vec_embeddings.npy')
    node_id_map = np.load('node2vec_node_id_map.npy', allow_pickle=True).item()
else:
    print("Υπολογισμός Node2Vec embeddings")
    node2vec = Node2Vec(G_train, dimensions=64, walk_length=20, num_walks=100, workers=4)
    n2v_model = node2vec.fit(window=10, min_count=1)

    #Δημιουργεία λεξικου node_id_map που αντιστοιχίζει 
    #κάθε κόμβο του γράφου σε έναν ακέραιο δείκτη.
    node_id_map = {node: idx for idx, node in enumerate(G_train.nodes())}
    # Δημιουργία πίνακα embeddings 64 διαστάσεων απο 0
    embedding_matrix = np.zeros((len(node_id_map), 64))
    # Γέμισμα του πίνακα με τα embeddings του Node2Vec
    for node, idx in node_id_map.items():
        embedding_matrix[idx] = n2v_model.wv[str(node)]
    
    np.save('node2vec_embeddings.npy', embedding_matrix)
    np.save('node2vec_node_id_map.npy', node_id_map)

# Δημιουργία PyTorch Geometric Data object
features = torch.tensor(embedding_matrix, dtype=torch.float)
edge_index = torch.tensor(list(G_train.edges())).t().contiguous()

edge_index = edge_index.type(torch.long)
index_map = {node: i for i, node in enumerate(G_train.nodes())}
edge_index = edge_index.apply_(lambda x: index_map[x])

pyg_data = Data(x=features, edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
# Εκπαίδευση GCN για να μάθει τα embeddings των κόμβων   
model = GCN(in_channels=64, hidden_channels=32, out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

prev_loss = float('inf')
for epoch in range(100000):
    optimizer.zero_grad()
    out = model(pyg_data.x, pyg_data.edge_index)
    loss = F.mse_loss(out, pyg_data.x)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    if epoch > 0 and loss.item() > prev_loss:
        print("stopping: loss increased.")
        break
    if (prev_loss - loss.item()) < 0.0001:
        print("stopping: loss below threshold.")
        break
    prev_loss = loss.item()
final_embeddings = out.detach().cpu().numpy()

def gcn_sim(u, v):
    i, j = node_id_map[u], node_id_map[v]
    return cosine_similarity([final_embeddings[i]], [final_embeddings[j]])[0][0]

def jaccard_sim(u, v):
    return next(jaccard_coefficient(G_train, [(u, v)]))[2]

def common_neighbors(u, v):
    return len(list(nx.common_neighbors(G_train, u, v)))

def adamic_adar(u, v):
    return list(nx.adamic_adar_index(G_train, [(u, v)]))[0][2]

def preferential_attachment(u, v):
    return list(nx.preferential_attachment(G_train, [(u, v)]))[0][2]


#B abstracts και υπολογισμό embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("abstracts.txt", "r", encoding="utf-8") as f:
    abstracts = [line.strip() for line in f]

#ΜpNet
mpnet_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
mpnet_model.to(device)

if os.path.exists("mpnet_embeddings.npy"):
    print("Loading cached MPNet embeddings")
    mpnet_emb = np.load("mpnet_embeddings.npy")
else:
    print("Computing MPNet embeddings")
    mpnet_emb = mpnet_model.encode(abstracts, show_progress_bar=True)
    np.save("mpnet_embeddings.npy", mpnet_emb)
    print("Saved mpnet_embeddings.npy")

# SBERT
if os.path.exists("sbert_emb.npy"):
    print("Loading cached SBERT embeddings...")
    sbert_emb = np.load("sbert_emb.npy")
else:
    print("Computing SBERT embeddings...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_emb = sbert_model.encode(abstracts, show_progress_bar=True)
    np.save("sbert_emb.npy", sbert_emb)
    print("Saved SBERT embeddings to sbert_emb.npy")

#tfidf
tfidf = TfidfVectorizer(max_features=5000)
tfidf_emb = tfidf.fit_transform(abstracts)

# SciBERT   
sci_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
sci_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)

if os.path.exists("scibert_emb.npy"):
    print("Loading cached SciBERT embeddings")
    scibert_emb = np.load("scibert_emb.npy")
else:
    print("Computing SciBERT embeddings")
    scibert_emb = []
    batch_size = 16
    sci_model.eval()
    with torch.no_grad():
        for i in range(0, len(abstracts), batch_size):
            batch = abstracts[i:i+batch_size]
            inputs = sci_tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = sci_model(**inputs)

            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()

            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            scibert_emb.append(mean_embeddings.cpu().numpy())

    scibert_emb = np.vstack(scibert_emb)
    np.save("scibert_emb.npy", scibert_emb)
    print("Saved SciBERT embeddings to scibert_emb.npy")


def preprocess(text):
    text = re.sub(r'\W+', ' ', text.lower())
    return text.split()

# Word2Vec
if os.path.exists("word2vec.model"):
    print("Loading cached Word2Vec model")
    w2v_model = Word2Vec.load("word2vec.model")
else:
    print("Training Word2Vec model")
    tokenized_abstracts = [preprocess(doc) for doc in abstracts]
    w2v_model = Word2Vec(sentences=tokenized_abstracts, vector_size=300, window=5, min_count=2, workers=4, epochs=30)
    w2v_model.save("word2vec.model")
    print("Saved Word2Vec model")

def get_w2v_vector(words, model):
    vectors = [model.wv[w] for w in words if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

if os.path.exists("word2vec_emb.npy"):
    print("Loading cached Word2Vec embeddings...")
    word2vec_emb = np.load("word2vec_emb.npy")
else:
    print("Computing Word2Vec abstract embeddings")
    word2vec_emb = np.array([get_w2v_vector(preprocess(doc), w2v_model) for doc in abstracts])
    np.save("word2vec_emb.npy", word2vec_emb)
    print("Saved word2vec_emb.npy")

def sbert_sim(u, v):
    return cosine_similarity([sbert_emb[u]], [sbert_emb[v]])[0][0]

def tfidf_sim(u, v):
    return cosine_similarity(tfidf_emb[u], tfidf_emb[v])[0][0]

def scibert_sim(u, v):
    return cosine_similarity([scibert_emb[u]], [scibert_emb[v]])[0][0]

def mpnet_sim(u, v):
    return cosine_similarity([mpnet_emb[u]], [mpnet_emb[v]])[0][0]

def word2vec_sim(u, v):
    return cosine_similarity([word2vec_emb[u]], [word2vec_emb[v]])[0][0]


#C authors features 

with open("authors.txt", "r", encoding="utf-8") as f:
    author_lines = f.readlines()

paper_authors = []
for line in author_lines:
    authors = line.strip().split("|--|")[1].split(",")
    paper_authors.append([a.strip() for a in authors if a.strip()])

G_authors = nx.Graph()
for authors in paper_authors:
    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            G_authors.add_edge(authors[i], authors[j])

if os.path.exists("author_embeddings.pkl"):
    print("Loading cached author embeddings...")
    with open("author_embeddings.pkl", "rb") as f:
        author_embeddings = pickle.load(f)
else:
    print("Computing author embeddings")
    node2vec_auth = Node2Vec(G_authors, dimensions=64, walk_length=10, num_walks=100, workers=4)
    model_auth = node2vec_auth.fit(window=10, min_count=1)

    author_embeddings = {a: model_auth.wv[a] for a in G_authors.nodes()}
    with open("author_embeddings.pkl", "wb") as f:
        pickle.dump(author_embeddings, f)
    print("Saved author embeddings")

#C1 Compute paper-level author embeddings
if os.path.exists("paper_author_emb.npy"):
    paper_author_emb = np.load("paper_author_emb.npy")
else:
    print("Computing paper-level author embeddings")
    paper_authors = []
    for line in author_lines:
        authors = line.strip().split("|--|")[1].split(",")
        authors = [a.strip() for a in authors if a.strip()]
        paper_authors.append(authors)

    paper_author_emb = []
    for authors in paper_authors:
        vecs = [author_embeddings[a] for a in authors if a in author_embeddings]
        if vecs:
            paper_author_emb.append(np.mean(vecs, axis=0))
        else:
            paper_author_emb.append(np.zeros(model_auth.vector_size))
    paper_author_emb = np.vstack(paper_author_emb)
    np.save("paper_author_emb.npy", paper_author_emb)
    print("Saved paper_author_emb.npy")

def author_sim(u, v):
    return cosine_similarity([paper_author_emb[u]], [paper_author_emb[v]])[0][0]


pairs = train[["source","target"]].values
labels = train["label"].values
filtered_pairs = [(u, v) for u, v in pairs if u in G_train and v in G_train]
filtered_labels = [labels[i] for i, (u, v) in enumerate(pairs) if u in G_train and v in G_train]

#D training 
def compute_features(filtered_pairs):
    X = []
    for u, v in filtered_pairs:
        feats = [
            author_sim(u, v),
            jaccard_sim(u, v),
            common_neighbors(u, v),
            adamic_adar(u, v),
            sbert_sim(u, v),
            tfidf_sim(u, v),
            scibert_sim(u, v),
            mpnet_sim(u, v),
            word2vec_sim(u, v),
            gcn_sim(u, v),
            preferential_attachment(u, v),

        ]
        X.append(feats)
    return np.array(X)

#Train
print("Computing train features...")
X_train = compute_features(filtered_pairs)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

final_model = XGBClassifier(
    learning_rate=0.05,
    max_depth=5,
    n_estimators=500,
    eval_metric='logloss',
    random_state=42
)

final_model.fit(X_train_scaled, filtered_labels)
train_probs = final_model.predict_proba(X_train_scaled)[:, 1]
train_logloss = log_loss(filtered_labels, train_probs)
print(f"Log Loss στο train set : {train_logloss:.5f}")

#Test prediction
print("Loading test data")
test_df = pd.read_csv("test.txt", header=None, names=["source", "target"])
test_pairs = test_df.values

print("Computing test features")
X_test = compute_features(test_pairs)
X_test_scaled = scaler.transform(X_test)
probs = final_model.predict_proba(X_test_scaled)[:, 1]

#Αποθήκευση submission
submission = test_df.copy()
submission["Label"] = probs
submission.reset_index(inplace=True)
submission = submission[["index", "Label"]]
submission.columns = ["ID", "Label"]
submission.to_csv("submission.csv", index=False)
print("saved submission.csv")
