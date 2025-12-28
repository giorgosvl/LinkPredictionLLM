# Link Prediction on a Scientific Article Graph

This project implements a **link prediction model** on a graph of scientific papers, where each node represents a paper and each edge indicates a relationship between papers. The goal is to predict the **probability that a link exists between two nodes**.

## Method
The solution combines **graph, text, and author features**:

### Graph-based features
Node2Vec embeddings, GCN-refined node embeddings (PyTorch Geometric),  Common Neighbors,  Jaccard Coefficient,  Adamic–Adar,  Preferential Attachment  

### Abstract similarity (cosine similarity)
MPNet,  SBERT,  SciBERT,  TF-IDF,  Word2Vec  

### Author similarity
Author collaboration graph + Node2Vec,  Aggregated paper-level embeddings  

All features are standardized and passed into an **XGBoost classifier**, which outputs the final link probability. Cached embeddings are stored to avoid recomputation. Final predictions are saved in **`submission.csv`**.

## Tech Stack
Python · PyTorch Geometric · NetworkX · Sentence Transformers · Transformers · Gensim · Scikit-learn · XGBoost · NumPy · Pandas
