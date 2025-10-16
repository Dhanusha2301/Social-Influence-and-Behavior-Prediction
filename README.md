# Social-Influence-and-Behavior-Prediction
This project predicts how behaviors and opinions spread in social networks by combining Graph Neural Networks (GraphSAGE and GAT) with causal inference techniques. It generates node embeddings to capture structural and feature information, while causal analysis identifies true influencers, separating genuine impact from mere correlation.
# Social Influence and Behavior Prediction

This project predicts how behaviors and opinions propagate in social networks using **Graph Neural Networks (GraphSAGE and GAT)** combined with **causal inference techniques**. It identifies **true influencers** by distinguishing genuine influence from correlation, enabling targeted interventions for marketing, health awareness campaigns, and policy adoption.

---

## Features

- Node embeddings with **GraphSAGE** and **Graph Attention Networks (GAT)**.
- Identification of **top influencers** using causal scoring.
- Prediction of behavior spread in social networks.
- Applicable for marketing, healthcare, and policy adoption interventions.
- Easy to extend for large real-world networks.

---

## Installation

### Prerequisites
Make sure Python 3.8+ is installed. Recommended packages:

```bash
pip install torch torch_geometric networkx scikit-learn matplotlib numpy
