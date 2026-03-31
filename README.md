# 🕸️ Anti-Money Laundering (AML) — Graph Analysis & Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyG](https://img.shields.io/badge/PyTorch_Geometric-GNN-purple) ![NetworkX](https://img.shields.io/badge/NetworkX-Graph-blue) ![License](https://img.shields.io/badge/License-MIT-green)

> Detecting money laundering patterns in financial transaction networks using Graph Neural Networks (GraphSAGE).

## Problem Statement

Money laundering moves **$800B–$2T annually** through the global financial system. Traditional rule-based systems miss **85-95%** of laundering activity. Graph-based ML models can capture the **relational structure** of suspicious transaction patterns that tabular models miss.

## Key AML Patterns Detected

| Pattern | Description |
|---------|-------------|
| **Fan-out** | 1 account → many accounts (layering/dispersion) |
| **Fan-in** | Many accounts → 1 account (consolidation) |
| **Cycle** | Circular transaction loops |
| **Scatter-Gather** | Combine fan-out + fan-in |
| **U-turn** | Funds go out and come back through intermediaries |

## Graph Architecture

```
Transactions DataFrame
    ↓
Graph Construction (NetworkX DiGraph)
Nodes = bank accounts | Edges = transactions (amount, currency, timestamp)
    ↓
Graph Features: PageRank, Betweenness Centrality, In/Out Degree, Clustering Coefficient
    ↓
PyTorch Geometric — GraphSAGE
(aggregates neighborhood information for each node)
    ↓
Node Classification: Laundering / Legitimate
```

## Kaggle Notebook

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ibrahimagabardiop/aml-graph-neural-network)

## Dataset

- [`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) — IBM synthetic AML (237 votes)

## Tech Stack

```
Python · PyTorch · PyTorch Geometric · NetworkX · scikit-learn · Matplotlib
```

## Author

**Ibrahima Gabar Diop** — [Kaggle](https://www.kaggle.com/ibrahimagabardiop) · [GitHub](https://github.com/Gblack98)
