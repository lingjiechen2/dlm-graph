"""
PyG 节点分类模型：GCN、GraphSAGE、GAT、GATv2。
接口与参考项目类似：堆叠消息传递层 + 节点级输出（无图级 readout）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv


def build_model(
    name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    dropout: float,
    heads: int = 8,
) -> nn.Module:
    name = name.lower()
    if name in ("gcn", "graphconv"):
        return GCNNodeClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name in ("sage", "graphsage", "graph_sage"):
        return GraphSAGENodeClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name == "gat":
        return GATNodeClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout, heads)
    if name in ("gatv2", "gat_v2"):
        return GATv2NodeClassifier(
            in_dim, hidden_dim, out_dim, num_layers, dropout, heads
        )
    raise ValueError(f"Unknown model: {name}. Choose gcn, sage, gat, gatv2.")


class GCNNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGENodeClassifier(nn.Module):
    """GraphSAGE：默认 mean 聚合，normalize=True。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(
                SAGEConv(dims[i], dims[i + 1], aggr="mean", normalize=True)
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATNodeClassifier(nn.Module):
    """多层 GAT：除最后一层外 multi-head concat，最后一层单头输出类别 logits。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        heads: int = 8,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.heads = heads
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout)
            )
            return

        self.convs.append(
            GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        )
        in_channels = hidden_dim * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels,
                    hidden_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            in_channels = hidden_dim * heads
        self.convs.append(
            GATConv(
                in_channels,
                out_dim,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATv2NodeClassifier(nn.Module):
    """多层 GATv2：除最后一层外 multi-head concat，最后一层单头输出类别 logits。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
        heads: int = 8,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.heads = heads
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATv2Conv(in_dim, out_dim, heads=1, concat=False, dropout=dropout)
            )
            return

        self.convs.append(
            GATv2Conv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        )
        in_channels = hidden_dim * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    in_channels,
                    hidden_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            in_channels = hidden_dim * heads
        self.convs.append(
            GATv2Conv(
                in_channels,
                out_dim,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
