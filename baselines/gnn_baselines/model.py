"""
PyG node-classification baselines.

Supported models:
- GCN
- GraphSAGE
- GIN
- GAT
- GATv2
- GraphTransformer
- MixHop
- DIFFormer
- SGFormer
- NodeFormer
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MixHopConv,
    SAGEConv,
    TransformerConv,
)
from torch_geometric.utils import degree


@dataclass
class GTConfig:
    hidden_dim: int
    num_layers: int
    n_heads: int
    dropout: float
    layer_norm: bool = False
    batch_norm: bool = False
    use_weight: bool = True
    use_graph: bool = True
    graph_weight: float = 0.8
    use_residual: bool = True
    use_source: bool = True
    use_act: bool = True
    alpha: float = 0.5
    kernel: str = "simple"
    aggregate: str = "add"
    kernel_trans: str = "softmax"
    projection_matrix_type: str | None = "a"
    nb_random_features: int = 30
    use_gumbel: bool = True
    nb_gumbel_sample: int = 10
    rb_order: int = 2
    rb_trans: str = "sigmoid"
    use_edge_loss: bool = True
    edge_loss_weight: float = 0.1
    tau: float = 0.25


def build_model(
    name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    dropout: float,
    heads: int = 8,
    gt_config: GTConfig | None = None,
) -> nn.Module:
    name = name.lower()
    if name in ("gcn", "graphconv"):
        return GCNNodeClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name in ("sage", "graphsage", "graph_sage"):
        return GraphSAGENodeClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name == "gin":
        return GINNodeClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name == "gat":
        return GATNodeClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout, heads)
    if name in ("gatv2", "gat_v2"):
        return GATv2NodeClassifier(
            in_dim, hidden_dim, out_dim, num_layers, dropout, heads
        )
    if name in ("graphtransformer", "graph_transformer"):
        return GraphTransformerNodeClassifier(
            in_dim, hidden_dim, out_dim, num_layers, dropout, heads
        )
    if name == "mixhop":
        return MixHopNodeClassifier(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name == "difformer":
        return DIFFormerNodeClassifier(
            in_dim, out_dim, gt_config or GTConfig(hidden_dim, num_layers, heads, dropout)
        )
    if name == "sgformer":
        return SGFormerNodeClassifier(
            in_dim, out_dim, gt_config or GTConfig(hidden_dim, num_layers, heads, dropout)
        )
    if name == "nodeformer":
        return NodeFormerNodeClassifier(
            in_dim, out_dim, gt_config or GTConfig(hidden_dim, num_layers, heads, dropout)
        )
    raise ValueError(
        "Unknown model: "
        f"{name}. Choose gcn, sage, gin, gat, gatv2, graphtransformer, mixhop, difformer, sgformer, nodeformer."
    )


def _build_sparse_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    values: torch.Tensor | None = None,
) -> torch.Tensor:
    if values is None:
        values = torch.ones(edge_index.shape[1], device=edge_index.device)
    indices = torch.stack([edge_index[1], edge_index[0]], dim=0)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def _sparse_matmul(
    edge_index: torch.Tensor,
    num_nodes: int,
    x: torch.Tensor,
    values: torch.Tensor | None = None,
) -> torch.Tensor:
    adj = _build_sparse_adj(edge_index, num_nodes, values)
    return torch.sparse.mm(adj, x)


def _build_exact_k_hop_edge_indices(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_hops: int,
) -> list[torch.Tensor]:
    if max_hops <= 0:
        return []

    edge_index_cpu = edge_index.detach().cpu()
    adj = [set() for _ in range(num_nodes)]
    for src, dst in edge_index_cpu.t().tolist():
        adj[src].add(dst)

    hop_edges: list[torch.Tensor] = []
    visited = [set([i]) for i in range(num_nodes)]
    frontier = [set(nei) for nei in adj]

    for hop in range(1, max_hops + 1):
        pairs = []
        for src in range(num_nodes):
            next_nodes = frontier[src] - visited[src]
            if not next_nodes:
                continue
            visited[src].update(next_nodes)
            pairs.extend((src, dst) for dst in next_nodes)
        if pairs:
            hop_tensor = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        else:
            hop_tensor = torch.empty((2, 0), dtype=torch.long)
        hop_edges.append(hop_tensor.to(edge_index.device))

        if hop == max_hops:
            break

        new_frontier = [set() for _ in range(num_nodes)]
        for src in range(num_nodes):
            for mid in frontier[src]:
                new_frontier[src].update(adj[mid])
        frontier = new_frontier

    return hop_edges


def _full_attention_conv(
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    kernel: str,
) -> torch.Tensor:
    if kernel == "simple":
        # Match OpenGT's implementation: normalize by the global tensor norm
        # instead of per-token/per-head norms.
        qs = qs / qs.norm(p=2).clamp_min(1e-12)
        ks = ks / ks.norm(p=2).clamp_min(1e-12)
        num_nodes = qs.shape[0]

        kvs = torch.einsum("nhm,nhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)
        attention_num += vs.sum(dim=0, keepdim=True).expand(num_nodes, -1, -1)

        ks_sum = ks.sum(dim=0)
        attention_den = torch.einsum("nhm,hm->nh", qs, ks_sum).unsqueeze(-1)
        attention_den += num_nodes
        return attention_num / attention_den.clamp_min(1e-12)

    if kernel == "sigmoid":
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks))
        attention_den = attention_num.sum(dim=1, keepdim=True)
        attention = attention_num / attention_den.clamp_min(1e-12)
        return torch.einsum("nlh,lhd->nhd", attention, vs)

    raise ValueError(f"Unsupported DIFFormer kernel: {kernel}")


def _gcn_headwise(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    num_nodes, num_heads = x.shape[0], x.shape[1]
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=x.dtype).clamp_min(1.0)
    norm = (1.0 / deg[col]).sqrt() * (1.0 / deg[row]).sqrt()
    out = []
    for head in range(num_heads):
        out.append(_sparse_matmul(edge_index, num_nodes, x[:, head], norm))
    return torch.stack(out, dim=1)


class GCNNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
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
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr="mean", normalize=True))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GINNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            mlp_hidden = hidden_dim if i < num_layers - 1 else max(hidden_dim, out_dim)
            mlp = nn.Sequential(
                nn.Linear(dims[i], mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, dims[i + 1]),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float, heads: int = 8):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout))
            return

        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))
        in_channels = hidden_dim * heads
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, concat=True, dropout=dropout))
            in_channels = hidden_dim * heads
        self.convs.append(GATConv(in_channels, out_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATv2NodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float, heads: int = 8):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATv2Conv(in_dim, out_dim, heads=1, concat=False, dropout=dropout))
            return

        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))
        in_channels = hidden_dim * heads
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(in_channels, hidden_dim, heads=heads, concat=True, dropout=dropout))
            in_channels = hidden_dim * heads
        self.convs.append(GATv2Conv(in_channels, out_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphTransformerNodeClassifier(nn.Module):
    """Lightweight graph transformer baseline using PyG TransformerConv."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float, heads: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        self.activation = nn.GELU()
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(TransformerConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout))
            return

        self.convs.append(TransformerConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))
        in_channels = hidden_dim * heads
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels, hidden_dim, heads=heads, concat=True, dropout=dropout))
            in_channels = hidden_dim * heads
        self.convs.append(TransformerConv(in_channels, out_dim, heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MixHopNodeClassifier(nn.Module):
    """MixHop node classifier with PyG MixHopConv using powers [0, 1, 2]."""

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
        self.powers = [0, 1, 2]
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(MixHopConv(in_dim, out_dim, powers=self.powers))
            self.out_proj = nn.Linear(out_dim * len(self.powers), out_dim)
            return

        self.convs.append(MixHopConv(in_dim, hidden_dim, powers=self.powers))
        in_channels = hidden_dim * len(self.powers)
        for _ in range(num_layers - 2):
            self.convs.append(MixHopConv(in_channels, hidden_dim, powers=self.powers))
            in_channels = hidden_dim * len(self.powers)
        self.convs.append(MixHopConv(in_channels, out_dim, powers=self.powers))
        self.out_proj = nn.Linear(out_dim * len(self.powers), out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if hasattr(self, "out_proj"):
            x = self.out_proj(x)
        return x


class DIFFormerConvLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: GTConfig):
        super().__init__()
        self.Wk = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wq = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wv = nn.Linear(dim_in, dim_out * config.n_heads) if config.use_weight else None
        self.norm = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(config.dropout)

        self.dim_out = dim_out
        self.n_heads = config.n_heads
        self.kernel = config.kernel
        self.use_graph = config.use_graph
        self.use_weight = config.use_weight
        self.graph_weight = config.graph_weight
        self.residual = config.use_residual
        self.use_source = config.use_source
        self.alpha = config.alpha
        self.use_bn = config.batch_norm

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        query = self.Wq(x).reshape(-1, self.n_heads, self.dim_out)
        key = self.Wk(x).reshape(-1, self.n_heads, self.dim_out)
        if self.use_weight:
            value = self.Wv(x).reshape(-1, self.n_heads, self.dim_out)
        else:
            value = x.reshape(-1, 1, self.dim_out)

        attn = _full_attention_conv(query, key, value, self.kernel)
        if self.use_graph:
            graph_out = _gcn_headwise(value, edge_index)
            if self.graph_weight > 0:
                out = (1 - self.graph_weight) * attn + self.graph_weight * graph_out
            else:
                out = attn + graph_out
        else:
            out = attn
        out = out.mean(dim=1)

        if self.use_source:
            out = out + x0
        if self.residual:
            out = out * self.alpha + x * (1 - self.alpha)
        if self.use_bn:
            out = self.norm(out)
        return self.dropout(out)


class DIFFormerNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: GTConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(in_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            DIFFormerConvLayer(config.hidden_dim, config.hidden_dim, config)
            for _ in range(config.num_layers)
        )
        self.classifier = nn.Linear(config.hidden_dim, out_dim)
        self.last_aux_loss = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x0 = x
        for layer in self.layers:
            x = layer(x, edge_index, x0)
        self.last_aux_loss = None
        return self.classifier(x)


class TransConvLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: GTConfig):
        super().__init__()
        self.Wk = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wq = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wv = nn.Linear(dim_in, dim_out * config.n_heads) if config.use_weight else None
        self.dim_out = dim_out
        self.num_heads = config.n_heads
        self.use_weight = config.use_weight
        self.residual = config.use_residual
        self.use_act = config.use_act
        if config.layer_norm:
            self.norm = nn.LayerNorm(dim_out)
        elif config.batch_norm:
            self.norm = nn.BatchNorm1d(dim_out)
        else:
            self.norm = None
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.Wq(x).reshape(-1, self.num_heads, self.dim_out)
        key = self.Wk(x).reshape(-1, self.num_heads, self.dim_out)
        if self.use_weight:
            value = self.Wv(x).reshape(-1, self.num_heads, self.dim_out)
        else:
            value = x.reshape(-1, 1, self.dim_out)

        out = _full_attention_conv(query, key, value, kernel="simple").mean(dim=1)
        if self.residual:
            out = out + x
        if self.norm is not None:
            out = self.norm(out)
        if self.use_act:
            out = self.activation(out)
        return self.dropout(out)


class SGFormerNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: GTConfig):
        super().__init__()
        self.config = config
        self.trans_input = nn.Linear(in_dim, config.hidden_dim)
        self.trans_layers = nn.ModuleList(
            TransConvLayer(config.hidden_dim, config.hidden_dim, config)
            for _ in range(config.num_layers)
        )
        self.use_graph = config.use_graph
        if self.use_graph:
            self.graph_input = nn.Linear(in_dim, config.hidden_dim)
            self.graph_conv = GCNConv(config.hidden_dim, config.hidden_dim)
            classifier_in = config.hidden_dim if config.aggregate == "add" else 2 * config.hidden_dim
        else:
            classifier_in = config.hidden_dim
        self.classifier = nn.Linear(classifier_in, out_dim)
        self.last_aux_loss = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        trans_x = self.trans_input(x)
        for layer in self.trans_layers:
            trans_x = layer(trans_x)

        if self.use_graph:
            graph_x = self.graph_input(x)
            graph_x = self.graph_conv(graph_x, edge_index)
            if self.config.aggregate == "add":
                out = self.config.graph_weight * graph_x + (1 - self.config.graph_weight) * trans_x
            elif self.config.aggregate == "cat":
                out = torch.cat([trans_x, graph_x], dim=-1)
            else:
                raise ValueError(f"Unsupported SGFormer aggregate: {self.config.aggregate}")
        else:
            out = trans_x

        self.last_aux_loss = None
        return self.classifier(out)


def _create_projection_matrix(m: int, d: int, seed: int = 0, scaling: int = 0) -> torch.Tensor:
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        q, _ = torch.linalg.qr(torch.randn((d, d)))
        block_list.append(q.t())
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        q, _ = torch.linalg.qr(torch.randn((d, d)))
        block_list.append(q.t()[:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}.")

    return torch.diag(multiplier) @ final_matrix


def _relu_kernel_transformation(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix: torch.Tensor | None = None,
    numerical_stabilizer: float = 0.001,
) -> torch.Tensor:
    del is_query
    if projection_matrix is None:
        return data.relu() + numerical_stabilizer
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32, device=data.device))
    data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    return data_dash.relu() + numerical_stabilizer


def _softmax_kernel_transformation(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix: torch.Tensor,
    numerical_stabilizer: float = 1e-6,
) -> torch.Tensor:
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32, device=data.device)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32, device=data.device))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data).sum(dim=-1, keepdim=True) / 2.0
    last_dim = len(data_dash.shape) - 1
    attn_dim = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dim, keepdim=True)[0])
            + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(torch.max(data_dash, dim=last_dim, keepdim=True)[0], dim=attn_dim, keepdim=True)[0]
            )
            + numerical_stabilizer
        )
    return data_dash


def _kernelized_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_transformation,
    projection_matrix: torch.Tensor | None,
    edge_index: torch.Tensor,
    tau: float = 0.25,
    return_weight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix)
    key_prime = kernel_transformation(key, False, projection_matrix)
    query_prime = query_prime.permute(1, 0, 2, 3)
    key_prime = key_prime.permute(1, 0, 2, 3)
    value = value.permute(1, 0, 2, 3)

    kvs = torch.einsum("nbhm,nbhd->bhmd", key_prime, value)
    z_num = torch.einsum("nbhm,bhmd->nbhd", query_prime, kvs)
    ks_sum = key_prime.sum(dim=0)
    z_den = torch.einsum("nbhm,bhm->nbh", query_prime, ks_sum).unsqueeze(-1)

    z_output = (z_num.permute(1, 0, 2, 3) / z_den.permute(1, 0, 2, 3).clamp_min(1e-12))

    if not return_weight:
        return z_output

    start, end = edge_index
    query_end, key_start = query_prime[end], key_prime[start]
    edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start).permute(1, 0, 2)
    attn_normalizer = torch.einsum("nbhm,bhm->nbh", query_prime, ks_sum)
    edge_attn_den = attn_normalizer[end].permute(1, 0, 2)
    return z_output, edge_attn_num / edge_attn_den.clamp_min(1e-12)


def _kernelized_gumbel_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_transformation,
    projection_matrix: torch.Tensor | None,
    edge_index: torch.Tensor,
    k_samples: int = 10,
    tau: float = 0.25,
    return_weight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix).permute(1, 0, 2, 3)
    key_prime = kernel_transformation(key, False, projection_matrix).permute(1, 0, 2, 3)
    value = value.permute(1, 0, 2, 3)

    gumbels = (
        -torch.empty(key_prime.shape[:-1] + (k_samples,), device=query.device).exponential_().log()
    ) / tau
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4)
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", key_t_gumbel, value)
    z_num = torch.einsum("nbhm,bhkmd->nbhkd", query_prime, kvs)
    ks_sum = key_t_gumbel.sum(dim=0)
    z_den = torch.einsum("nbhm,bhkm->nbhk", query_prime, ks_sum).unsqueeze(-1)
    z_output = torch.mean(z_num.permute(1, 0, 2, 3, 4) / z_den.permute(1, 0, 2, 3, 4).clamp_min(1e-12), dim=3)

    if not return_weight:
        return z_output

    start, end = edge_index
    query_end, key_start = query_prime[end], key_prime[start]
    edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start).permute(1, 0, 2)
    attn_normalizer = torch.einsum("nbhm,bhm->nbh", query_prime, key_prime.sum(dim=0))
    edge_attn_den = attn_normalizer[end].permute(1, 0, 2)
    return z_output, edge_attn_num / edge_attn_den.clamp_min(1e-12)


def _add_conv_relational_bias(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    b: torch.Tensor,
    rb_trans: str = "sigmoid",
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected x to have shape [B, N, H, D], got {tuple(x.shape)}")
    if x.shape[0] != 1:
        raise ValueError(
            "NodeFormer relational bias in this lightweight runner only supports B=1 "
            f"(single transductive graph), got batch size {x.shape[0]}"
        )

    row, col = edge_index
    num_nodes = x.shape[1]
    deg_in = degree(col, num_nodes, dtype=x.dtype).clamp_min(1.0)
    deg_out = degree(row, num_nodes, dtype=x.dtype).clamp_min(1.0)
    conv_output = []
    for i in range(x.shape[2]):
        if rb_trans == "sigmoid":
            b_i = b[i].sigmoid()
        elif rb_trans == "identity":
            b_i = b[i]
        else:
            raise NotImplementedError
        values = torch.ones_like(row, dtype=x.dtype) * b_i * (1.0 / deg_in[col]).sqrt() * (1.0 / deg_out[row]).sqrt()
        head_x = x[0, :, i, :]  # [N, D]
        conv_output.append(_sparse_matmul(edge_index, num_nodes, head_x, values))
    return torch.stack(conv_output, dim=1).unsqueeze(0)  # [1, N, H, D]


class NodeFormerConvLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: GTConfig):
        super().__init__()
        self.Wk = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wq = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wv = nn.Linear(dim_in, dim_out * config.n_heads)
        self.Wo = nn.Linear(dim_out * config.n_heads, dim_out)
        if config.rb_order >= 1:
            self.b = nn.Parameter(torch.FloatTensor(config.rb_order, config.n_heads), requires_grad=True)
            init_val = 0.1 if config.rb_trans == "sigmoid" else 1.0
            nn.init.constant_(self.b, init_val)
        else:
            self.b = None

        self.dim_out = dim_out
        self.n_heads = config.n_heads
        self.kernel_transformation = (
            _softmax_kernel_transformation if config.kernel_trans == "softmax" else _relu_kernel_transformation
        )
        self.norm = nn.LayerNorm(config.hidden_dim) if config.batch_norm else None
        self.config = config

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = x.unsqueeze(0)
        num_nodes = z.size(1)
        query = self.Wq(z).reshape(-1, num_nodes, self.n_heads, self.dim_out)
        key = self.Wk(z).reshape(-1, num_nodes, self.n_heads, self.dim_out)
        value = self.Wv(z).reshape(-1, num_nodes, self.n_heads, self.dim_out)

        if self.config.projection_matrix_type is None:
            projection_matrix = None
        else:
            dim = query.shape[-1]
            seed = int(torch.ceil(torch.abs(torch.sum(query) * 1e8)).item())
            projection_matrix = _create_projection_matrix(
                self.config.nb_random_features, dim, seed=seed
            ).to(query.device)

        if self.config.use_gumbel and self.training:
            z_next, weight = _kernelized_gumbel_softmax(
                query,
                key,
                value,
                self.kernel_transformation,
                projection_matrix,
                adjs[0],
                self.config.nb_gumbel_sample,
                self.config.tau,
                True,
            )
        else:
            z_next, weight = _kernelized_softmax(
                query,
                key,
                value,
                self.kernel_transformation,
                projection_matrix,
                adjs[0],
                self.config.tau,
                True,
            )

        if self.config.rb_order >= 1 and self.b is not None:
            for i in range(min(self.config.rb_order, len(adjs))):
                z_next += _add_conv_relational_bias(value, adjs[i], self.b[i], self.config.rb_trans)

        z_next = self.Wo(z_next.flatten(-2, -1))
        if self.config.use_residual:
            z_next += z
        if self.norm is not None:
            z_next = self.norm(z_next)
        if self.config.use_act:
            z_next = F.elu(z_next)
        z_next = F.dropout(z_next, p=self.config.dropout, training=self.training)

        edge_loss = None
        if self.config.use_edge_loss:
            row, col = adjs[0]
            d_in = degree(col, query.shape[1], dtype=query.dtype).clamp_min(1.0)
            d_norm = (1.0 / d_in[col]).reshape(1, -1, 1).repeat(1, 1, weight.shape[-1])
            edge_loss = torch.mean(weight.clamp_min(1e-12).log() * d_norm)

        return z_next.squeeze(0), edge_loss


class NodeFormerNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, config: GTConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(in_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            NodeFormerConvLayer(config.hidden_dim, config.hidden_dim, config)
            for _ in range(config.num_layers)
        )
        self.classifier = nn.Linear(config.hidden_dim, out_dim)
        self._cached_adjs: list[torch.Tensor] | None = None
        self._cached_signature: tuple[int, int, torch.device] | None = None
        self.last_aux_loss = None

    def _get_adjs(self, edge_index: torch.Tensor, num_nodes: int) -> list[torch.Tensor]:
        signature = (num_nodes, edge_index.shape[1], edge_index.device)
        if self._cached_adjs is None or self._cached_signature != signature:
            self._cached_adjs = _build_exact_k_hop_edge_indices(
                edge_index, num_nodes, max(self.config.rb_order, 1)
            )
            self._cached_signature = signature
        return self._cached_adjs

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        adjs = self._get_adjs(edge_index, x.shape[0])

        extra_loss = None
        for layer in self.layers:
            x, layer_loss = layer(x, adjs)
            if layer_loss is not None:
                extra_loss = layer_loss if extra_loss is None else extra_loss + layer_loss

        if extra_loss is not None and self.config.use_edge_loss:
            self.last_aux_loss = self.config.edge_loss_weight * extra_loss / self.config.num_layers
        else:
            self.last_aux_loss = None
        return self.classifier(x)
