"""
Graph Neural Network for the Economic Complexity product space.

Implements GraphSAGE for:
  1. Node regression  — predict a country's Economic Complexity Index (ECI)
  2. Link prediction  — predict which new products a country will export next

References
----------
Hamilton et al. (2017) Inductive Representation Learning on Large Graphs
https://arxiv.org/abs/1706.02216
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductSpaceGNN(nn.Module):
    """
    GraphSAGE encoder over the product-space bipartite graph.

    Nodes: countries + products.
    Edges: Revealed Comparative Advantage (RCA) links — country exports product
           competitively (RCA > 1). Edge weights = normalised RCA value.

    Args:
        in_channels:     input node feature dim
        hidden_channels: width of hidden layers
        out_channels:    embedding dimensionality
        num_layers:      number of message-passing layers
        dropout:         dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise ImportError(
                "torch-geometric is required for the GNN model. "
                "Install with: pip install torch-geometric"
            ) from exc

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.norms.append(nn.LayerNorm(dims[i + 1]))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # L2-norm for dot-product similarity

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode(x, edge_index), p=2, dim=-1)


class ECIRegressor(nn.Module):
    """
    Predicts a country's Economic Complexity Index from its GNN embedding.

    Used for supervised regression against World Bank / Atlas ECI values.
    """

    def __init__(self, gnn: ProductSpaceGNN, embedding_dim: int = 64) -> None:
        super().__init__()
        self.gnn = gnn
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        country_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.gnn(x, edge_index)
        return self.head(emb[country_mask]).squeeze(-1)


class LinkPredictor(nn.Module):
    """
    Predicts whether a country will form a new edge with a product node
    (i.e., start exporting it with RCA > 1 in the next period).

    Uses an MLP on the concatenated embeddings of the source (country)
    and destination (product) nodes.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        src = embeddings[src_idx]
        dst = embeddings[dst_idx]
        return self.mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)
