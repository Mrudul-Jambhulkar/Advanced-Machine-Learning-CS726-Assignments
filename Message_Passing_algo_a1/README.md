# Message Passing Inference in Graphical Models

This repository contains an implementation of a **Message Passing Algorithm** for inference in undirected graphical models, written entirely in Python using only standard libraries.

The primary goal of the project is to compute the **partition function (Z value)** of a probabilistic graphical model using the **Junction Tree Algorithm**, which involves the following core steps:

- Triangulation of the graph
- Extraction of maximal cliques
- Construction of a Junction Tree
- Assignment of potentials to cliques
- Message passing for marginal inference

## 📁 File

- `Assignment_1_AML.ipynb`: The main code file (originally created in Google Colab) containing the full pipeline from input parsing to partition function computation.

## 🚀 Features

- Graph representation using adjacency lists
- Detection and processing of **simplicial nodes**
- **Triangulation** using minimal degree heuristic
- **Clique extraction** and Junction Tree construction via maximum spanning tree over clique intersections
- **Potential assignment** and factor multiplication
- **Sum-product message passing** for computing the normalization constant \( Z \)


