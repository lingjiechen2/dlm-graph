#!/usr/bin/env python3
"""
Run PyG GCN, GraphSAGE, GIN, GAT, GATv2, MixHop, and several GT baselines
for node classification on Cora / PubMed.

By default, load processed_data.pt from the LLaGA repository's dataset
directory, matching the LLaGA preprocessing pipeline.
Use --source planetoid to switch to the official PyG Planetoid data.

Examples:
  python run_gnn.py --dataset cora --model gcn --lr 0.01 --num_layers 2 --hidden_dim 16
  python run_gnn.py --dataset pubmed --model sage --epochs 200
  python run_gnn.py --dataset cora --model gin --hidden_dim 128 --num_layers 2
  python run_gnn.py --dataset cora --model gat --heads 8 --hidden_dim 8
  python run_gnn.py --dataset cora --model gatv2 --heads 8 --hidden_dim 8
  python run_gnn.py --dataset cora --model mixhop --hidden_dim 64 --num_layers 2
  python run_gnn.py --dataset cora --model graphtransformer --hidden_dim 96 --heads 2 --lr 5e-4
  python run_gnn.py --dataset cora --model difformer --hidden_dim 96 --heads 1 --num_layers 3 --lr 5e-4
  python run_gnn.py --dataset cora --model sgformer --hidden_dim 96 --heads 1 --num_layers 2 --lr 5e-4
  python run_gnn.py --dataset cora --model nodeformer --hidden_dim 96 --heads 1 --num_layers 3 --lr 5e-4
  python run_gnn.py --source planetoid --dataset cora --data_root ./data
  python run_gnn.py --dataset cora --model gcn --gpu 1
  python run_gnn.py --dataset pubmed --model gcn --batch_size 1024 --neighbor_fanout 25

Results are saved by default to gnn/result/<timestamp>_<dataset>_<model>/
(metrics.json, best_model.pt).

batch_size:
  - 0 (default): run one full-graph forward pass without mini-batches.
  - >0: build NeighborLoader subgraph batches from training nodes; validation
    and testing still use the full graph.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

from model import GTConfig, build_model

FULL_GRAPH_ONLY_MODELS = {"difformer", "sgformer", "nodeformer"}
GT_MODELS = {"graphtransformer", *FULL_GRAPH_ONLY_MODELS}


def str2bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_gt_config(args: argparse.Namespace) -> GTConfig:
    return GTConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_heads=args.heads,
        dropout=args.dropout,
        layer_norm=bool(args.gt_layer_norm),
        batch_norm=bool(args.gt_batch_norm),
        use_weight=bool(args.gt_use_weight),
        use_graph=bool(args.gt_use_graph),
        graph_weight=args.gt_graph_weight,
        use_residual=bool(args.gt_use_residual),
        use_source=bool(args.gt_use_source),
        use_act=bool(args.gt_use_act),
        alpha=args.gt_alpha,
        kernel=args.gt_kernel,
        aggregate=args.gt_aggregate,
        kernel_trans=args.gt_kernel_trans,
        projection_matrix_type=args.gt_projection_matrix_type,
        nb_random_features=args.gt_nb_random_features,
        use_gumbel=bool(args.gt_use_gumbel),
        nb_gumbel_sample=args.gt_nb_gumbel_sample,
        rb_order=args.gt_rb_order,
        rb_trans=args.gt_rb_trans,
        use_edge_loss=bool(args.gt_use_edge_loss),
        edge_loss_weight=args.gt_edge_loss_weight,
        tau=args.gt_tau,
    )


@dataclass
class GraphDatasetInfo:
    """Metadata needed for node classification, compatible with Planetoid and LLaGA Data."""

    data: Data
    num_node_features: int
    num_classes: int
    source: str


def ensure_data_contiguous(data: Data) -> Data:
    """Make PyG tensors contiguous before NeighborLoader builds CSC structures."""
    for attr in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask"):
        if hasattr(data, attr):
            value = getattr(data, attr)
            if torch.is_tensor(value):
                setattr(data, attr, value.contiguous())
    return data


def clone_node_classification_tensors(data: Data) -> Data:
    """Keep only tensor attributes NeighborLoader can safely slice."""
    return Data(
        x=data.x.contiguous(),
        edge_index=data.edge_index.contiguous(),
        y=data.y.contiguous(),
        train_mask=data.train_mask.contiguous(),
        val_mask=data.val_mask.contiguous(),
        test_mask=data.test_mask.contiguous(),
    )


def _default_llaga_dataset_root() -> Path:
    # .../LLaGA/gnn/run_gnn.py -> .../LLaGA/dataset
    return Path(__file__).resolve().parent.parent / "dataset"


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _neighbor_sampling_backend_available() -> bool:
    """NeighborLoader requires either torch-sparse or pyg-lib as a backend."""
    return importlib.util.find_spec("torch_sparse") is not None or importlib.util.find_spec(
        "pyg_lib"
    ) is not None


def load_llaga_processed(dataset_name: str, dataset_root: Path) -> GraphDatasetInfo:
    """Load PyG Data from LLaGA's dataset/<name>/processed_data.pt."""
    name = dataset_name.lower()
    if name not in ("cora", "pubmed"):
        raise ValueError(
            f"The LLaGA source currently supports only cora and pubmed; got {dataset_name}. "
            "Use --source planetoid for CiteSeer."
        )
    path = dataset_root / name / "processed_data.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"LLaGA data file not found: {path}\n"
            f"Make sure processed_data.pt exists, or set the dataset directory with --llaga_dataset_root."
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, Data):
        raise TypeError(f"Expected torch_geometric.data.Data, got {type(data)}")

    for mask_name in ("train_mask", "val_mask", "test_mask"):
        if not hasattr(data, mask_name):
            raise AttributeError(f"Data is missing {mask_name}")
        m = getattr(data, mask_name)
        if m.dtype != torch.bool:
            setattr(data, mask_name, m.bool())

    if data.y.dim() > 1:
        data.y = data.y.view(-1)
    ensure_data_contiguous(data)

    n_feat = int(data.x.shape[1])
    if hasattr(data, "num_classes") and data.num_classes is not None:
        n_cls = int(data.num_classes)
    else:
        n_cls = int(data.y.max().item()) + 1

    return GraphDatasetInfo(
        data=data,
        num_node_features=n_feat,
        num_classes=n_cls,
        source=f"llaga:{path}",
    )


def load_planetoid(name: str, root: str) -> GraphDatasetInfo:
    name_map = {"cora": "Cora", "pubmed": "PubMed", "citeseer": "CiteSeer"}
    key = name.lower()
    if key not in name_map:
        raise ValueError(f"dataset must be cora / pubmed / citeseer; got {name}")
    dataset = Planetoid(root=root, name=name_map[key])
    data = ensure_data_contiguous(dataset[0])
    return GraphDatasetInfo(
        data=data,
        num_node_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        source=f"planetoid:{root}/{name_map[key]}",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None, gpu: int | None) -> torch.device:
    """
    Resolve the training device.
    - --device cpu: force CPU.
    - --device cuda or cuda:N: use the given device string; explicit cuda:N
      takes precedence over --gpu.
    - --gpu K: use cuda:K when no indexed cuda:N device is specified.
    - If neither is specified, use cuda:0 when CUDA is available; otherwise CPU.
    """
    if device is not None and device.lower() == "cpu":
        return torch.device("cpu")

    if device is not None and ":" in device and device.lower().startswith("cuda"):
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"{device} was specified, but CUDA is not available.")
        idx = dev.index if dev.index is not None else 0
        if torch.cuda.is_available() and (idx < 0 or idx >= torch.cuda.device_count()):
            raise RuntimeError(
                f"Invalid device {device}; visible GPU count is {torch.cuda.device_count()}."
            )
        return dev

    if gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu was set, but CUDA is not available.")
        n = torch.cuda.device_count()
        if gpu < 0 or gpu >= n:
            raise RuntimeError(f"--gpu={gpu} is invalid; visible GPU count is {n}.")
        return torch.device(f"cuda:{gpu}")

    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=1)
    correct = (pred == y[mask]).sum().item()
    return correct / int(mask.sum())


def _args_to_jsonable(args: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def save_run_results(
    result_base: Path,
    args: argparse.Namespace,
    info: GraphDatasetInfo,
    best_val: float,
    test_acc_at_best: float,
    final_test: float,
    best_state: dict | None,
    best_epoch: int,
    stopped_epoch: int | None,
    epochs_ran: int,
) -> Path:
    """Create the run directory under result_base and write metrics.json and best_model.pt."""
    result_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = result_base / f"{stamp}_{args.dataset}_{args.model}"
    run_dir.mkdir(parents=True)

    metrics = {
        "timestamp": stamp,
        "data_source": info.source,
        "dataset": args.dataset,
        "model": args.model,
        "source": args.source,
        "best_val_acc": best_val,
        "test_acc_at_best_val": test_acc_at_best,
        "final_test_acc": final_test,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "epochs_ran": epochs_ran,
        "early_stopped": stopped_epoch is not None,
        "num_node_features": info.num_node_features,
        "num_classes": info.num_classes,
        "hyperparameters": _args_to_jsonable(args),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if best_state is not None:
        torch.save(
            {
                "state_dict": best_state,
                "model": args.model,
                "num_node_features": info.num_node_features,
                "num_classes": info.num_classes,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "heads": args.heads,
                "metrics": {
                    "best_val_acc": best_val,
                    "test_acc_at_best_val": test_acc_at_best,
                    "final_test_acc": final_test,
                    "best_epoch": best_epoch,
                    "stopped_epoch": stopped_epoch,
                    "epochs_ran": epochs_ran,
                    "early_stopped": stopped_epoch is not None,
                },
                "hyperparameters": _args_to_jsonable(args),
                "data_source": info.source,
            },
            run_dir / "best_model.pt",
        )

    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PyG node classification "
            "(GCN / GraphSAGE / GIN / GAT / GATv2 / GraphTransformer / MixHop / DIFFormer / SGFormer / NodeFormer)"
        )
    )
    p.add_argument(
        "--source",
        type=str,
        default="llaga",
        choices=("llaga", "planetoid"),
        help="llaga: use LLaGA/dataset/*/processed_data.pt; planetoid: use official PyG data",
    )
    p.add_argument("--dataset", type=str, default="cora", help="cora | pubmed (citeseer is also available with planetoid)")
    p.add_argument(
        "--llaga_dataset_root",
        type=str,
        default=None,
        help="LLaGA dataset directory; defaults to this repository's LLaGA/dataset",
    )
    p.add_argument("--data_root", type=str, default="./data", help="planetoid only: data download directory")
    p.add_argument(
        "--model",
        type=str,
        default="gcn",
        help="gcn | sage | gin | gat | gatv2 | graphtransformer | mixhop | difformer | sgformer | nodeformer",
    )
    p.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension; for GAT/GATv2, this is the per-head output dimension",
    )
    p.add_argument("--num_layers", type=int, default=2, help="Number of message-passing layers")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument(
        "--heads", type=int, default=8, help="Number of GAT/GATv2 attention heads (gat/gatv2 only)"
    )
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early-stopping patience; 0 disables it, >0 stops after N epochs without val acc improvement",
    )
    p.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Minimum val acc improvement required; used only for early stopping and best checkpoint selection",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Training batch size: 0=full graph (default); >0 uses NeighborLoader if torch-sparse or pyg-lib is available, otherwise falls back to full graph",
    )
    p.add_argument(
        "--neighbor_fanout",
        type=int,
        default=25,
        help="Mini-batch only: number of neighbors sampled per layer, repeated num_layers times",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index, e.g. 0 or 1; equivalent to cuda:<index>, alternative to --device cuda:N",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu / cuda / cuda:N; defaults to GPU0 or CPU. Explicit cuda:N takes precedence over --gpu",
    )
    p.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Directory for saved results; defaults to result/ under this directory",
    )
    p.add_argument("--gt_use_graph", type=str2bool, default=True, help="Whether GT models fuse the graph-structure branch")
    p.add_argument("--gt_graph_weight", type=float, default=0.8, help="Weight for the GT graph branch")
    p.add_argument("--gt_use_weight", type=str2bool, default=True, help="Whether GT attention learns the value projection")
    p.add_argument("--gt_use_residual", type=str2bool, default=True, help="Whether GT models enable residual connections")
    p.add_argument("--gt_use_source", type=str2bool, default=True, help="Whether DIFFormer fuses the initial features")
    p.add_argument("--gt_use_act", type=str2bool, default=True, help="Whether GT models apply activation after each layer")
    p.add_argument("--gt_aggregate", type=str, default="add", help="SGFormer aggregation mode: add | cat")
    p.add_argument("--gt_kernel", type=str, default="simple", help="DIFFormer kernel: simple | sigmoid")
    p.add_argument("--gt_kernel_trans", type=str, default="softmax", help="NodeFormer kernel_trans: softmax | relu")
    p.add_argument("--gt_use_gumbel", type=str2bool, default=True, help="Whether NodeFormer uses Gumbel during training")
    p.add_argument("--gt_nb_gumbel_sample", type=int, default=10, help="Number of NodeFormer Gumbel samples")
    p.add_argument("--gt_nb_random_features", type=int, default=30, help="Number of NodeFormer random features")
    p.add_argument("--gt_rb_order", type=int, default=2, help="NodeFormer relational-bias order")
    p.add_argument("--gt_rb_trans", type=str, default="sigmoid", help="NodeFormer relational-bias transform")
    p.add_argument("--gt_use_edge_loss", type=str2bool, default=True, help="Whether NodeFormer enables edge loss")
    p.add_argument("--gt_edge_loss_weight", type=float, default=0.1, help="NodeFormer edge-loss weight")
    p.add_argument("--gt_projection_matrix_type", type=str, default="a", help="NodeFormer projection-matrix type; none disables it")
    p.add_argument("--gt_alpha", type=float, default=0.5, help="DIFFormer residual alpha")
    p.add_argument("--gt_layer_norm", type=str2bool, default=False, help="Whether GT models use layer norm")
    p.add_argument("--gt_batch_norm", type=str2bool, default=False, help="Whether GT models use batch norm")
    p.add_argument("--gt_tau", type=float, default=0.25, help="NodeFormer / kernelized attention temperature")
    p.add_argument("--no_save", action="store_true", help="Do not save metrics or checkpoint")
    return p.parse_args()


def run_gnn(args: argparse.Namespace | None = None) -> float:
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")
    if args.min_delta < 0:
        raise ValueError("--min_delta must be >= 0")

    device = resolve_device(args.device, args.gpu)
    print(f"device={device}")

    if args.source == "llaga":
        root = (
            Path(args.llaga_dataset_root).resolve()
            if args.llaga_dataset_root
            else _default_llaga_dataset_root()
        )
        info = load_llaga_processed(args.dataset, root)
    else:
        info = load_planetoid(args.dataset, args.data_root)

    if args.gt_projection_matrix_type is not None and str(args.gt_projection_matrix_type).lower() == "none":
        args.gt_projection_matrix_type = None

    use_mini_batch_requested = args.batch_size > 0
    model_name = args.model.lower()
    if model_name in FULL_GRAPH_ONLY_MODELS and use_mini_batch_requested:
        print(
            f"Warning: {args.model} currently supports only full-graph training "
            "in the OpenGT-style implementation; ignoring --batch_size and using full-graph training."
        )
        use_mini_batch_requested = False
    need_cpu_copy = use_mini_batch_requested and _neighbor_sampling_backend_available()
    if need_cpu_copy:
        # NeighborLoader slices every attribute; text/list metadata in LLaGA Data is not sampler-friendly.
        data_cpu = clone_node_classification_tensors(info.data)
    else:
        data_cpu = None
    data = info.data.to(device)

    gt_config = build_gt_config(args) if model_name in GT_MODELS else None
    model = build_model(
        name=args.model,
        in_dim=info.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=info.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
        gt_config=gt_config,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("-inf")
    best_state = None
    test_acc_at_best = 0.0
    best_epoch = 0
    stopped_epoch = None
    epochs_without_improvement = 0
    epochs_ran = 0

    use_mini_batch = use_mini_batch_requested and _neighbor_sampling_backend_available()
    train_loader = None
    if use_mini_batch_requested and not use_mini_batch:
        print(
            "torch-sparse or pyg-lib is not installed, so NeighborLoader cannot be used; "
            "ignoring --batch_size and using full-graph training. Try: pip install torch-sparse or pyg-lib"
        )

    if use_mini_batch:
        assert data_cpu is not None
        train_idx = data_cpu.train_mask.nonzero(as_tuple=False).view(-1)
        num_neighbors = [args.neighbor_fanout] * args.num_layers
        train_loader = NeighborLoader(
            data_cpu,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            input_nodes=train_idx,
            shuffle=True,
            num_workers=0,
        )
        print(
            f"mini-batch training: batch_size={args.batch_size}, "
            f"num_neighbors={num_neighbors}, train_nodes={train_idx.numel()}"
        )
    else:
        print("full-graph training (batch_size=0)")

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        model.train()
        if use_mini_batch:
            total_loss = 0.0
            n_train = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                bs = batch.batch_size
                loss = F.cross_entropy(out[:bs], batch.y[:bs])
                aux_loss = getattr(model, "last_aux_loss", None)
                if aux_loss is not None:
                    loss = loss + aux_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bs
                n_train += bs
            loss_item = total_loss / max(n_train, 1)
        else:
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            aux_loss = getattr(model, "last_aux_loss", None)
            if aux_loss is not None:
                loss = loss + aux_loss
            loss.backward()
            optimizer.step()
            loss_item = loss.item()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)
            test_acc = accuracy(logits, data.y, data.test_mask)

        if val_acc > best_val + args.min_delta:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            test_acc_at_best = test_acc
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            train_acc = accuracy(logits, data.y, data.train_mask)
            print(
                f"Epoch {epoch:03d} | loss {loss_item:.4f} | "
                f"train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f}"
            )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            stopped_epoch = epoch
            print(
                f"Early stopping at epoch {epoch}: val_acc did not improve by "
                f"at least {args.min_delta:g} for {args.patience} consecutive epochs. "
                f"Best epoch={best_epoch}, best_val_acc={best_val:.4f}."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        final_test = accuracy(logits, data.y, data.test_mask)

    print("-" * 60)
    print(f"data_source={info.source}")
    print(
        f"Dataset={args.dataset} | model={args.model} | "
        f"best_epoch={best_epoch} | epochs_ran={epochs_ran} | "
        f"best_val_acc={best_val:.4f} | test@best_val={test_acc_at_best:.4f} | "
        f"final_test_acc={final_test:.4f}"
    )

    if not args.no_save:
        out_base = Path(args.result_dir).resolve() if args.result_dir else _default_result_dir()
        run_dir = save_run_results(
            out_base,
            args,
            info,
            best_val,
            test_acc_at_best,
            final_test,
            best_state,
            best_epoch,
            stopped_epoch,
            epochs_ran,
        )
        print(f"Saved results to: {run_dir}")

    return float(test_acc_at_best)


if __name__ == "__main__":
    run_gnn()
