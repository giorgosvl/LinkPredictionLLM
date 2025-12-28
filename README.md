Link Prediction on a Scientific Article Graph
This project implements a link prediction model on a graph of scientific papers, where each node represents a paper and an edge indicates a relationship between two papers. The goal is to predict the probability that an edge exists between two nodes.
Approach (Feature Engineering + ML)
We combine graph-based, text-based, and author-based features:
Graph structure
Node2Vec embeddings
GCN-refined node embeddings (PyTorch Geometric)
Common Neighbors, Jaccard, Adamic–Adar, Preferential Attachment
Abstract similarity
MPNet
SBERT
SciBERT
TF-IDF
Word2Vec
→ cosine similarity per paper pair
Author similarity
Author collaboration graph + Node2Vec
Aggregated paper-level embeddings
All features are standardized and fed into an XGBoost classifier, which outputs the probability of a link. Cached embeddings are stored to avoid recomputation. The final predictions are written to submission.csv.
Tech Stack
Python • PyTorch Geometric • NetworkX • Sentence Transformers • Transformers • Gensim • Scikit-learn • XGBoost • NumPy • Pandas
