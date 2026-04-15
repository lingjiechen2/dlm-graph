#!/usr/bin/env python3
"""
在 Cora / PubMed 上做 PyG GCN、GraphSAGE、GAT、GATv2 节点分类。

默认从 LLaGA 仓库的 dataset 目录加载 processed_data.pt（与 LLaGA 预处理一致）。
可选 --source planetoid 使用 PyG 官方 Planetoid 数据。

示例:
  python run_gnn.py --dataset cora --model gcn --lr 0.01 --num_layers 2 --hidden_dim 16
  python run_gnn.py --dataset pubmed --model sage --epochs 200
  python run_gnn.py --dataset cora --model gat --heads 8 --hidden_dim 8
  python run_gnn.py --dataset cora --model gatv2 --heads 8 --hidden_dim 8
  python run_gnn.py --source planetoid --dataset cora --data_root ./data
  python run_gnn.py --dataset cora --model gcn --gpu 1
  python run_gnn.py --dataset pubmed --model gcn --batch_size 1024 --neighbor_fanout 25

结果默认保存到 gnn/result/<时间戳>_<dataset>_<model>/（metrics.json、best_model.pt）。

batch_size:
  - 0（默认）：全图一次前向，无 mini-batch。
  - >0：按训练节点做 NeighborLoader 子图 batch；验证/测试仍为全图。
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

from model import build_model


@dataclass
class GraphDatasetInfo:
    """节点分类所需元信息（兼容 Planetoid 与 LLaGA Data）。"""

    data: Data
    num_node_features: int
    num_classes: int
    source: str


def _default_llaga_dataset_root() -> Path:
    # .../LLaGA/gnn/run_gnn.py -> .../LLaGA/dataset
    return Path(__file__).resolve().parent.parent / "dataset"


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "result"


def _neighbor_sampling_backend_available() -> bool:
    """NeighborLoader 底层需要 torch-sparse 或 pyg-lib 之一。"""
    return importlib.util.find_spec("torch_sparse") is not None or importlib.util.find_spec(
        "pyg_lib"
    ) is not None


def load_llaga_processed(dataset_name: str, dataset_root: Path) -> GraphDatasetInfo:
    """从 LLaGA 的 dataset/<name>/processed_data.pt 加载 PyG Data。"""
    name = dataset_name.lower()
    if name not in ("cora", "pubmed"):
        raise ValueError(
            f"LLaGA 源目前仅支持 cora、pubmed，got {dataset_name}. "
            "CiteSeer 请使用 --source planetoid。"
        )
    path = dataset_root / name / "processed_data.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 LLaGA 数据文件: {path}\n"
            f"请确认已放置 processed_data.pt，或通过 --llaga_dataset_root 指定 dataset 目录。"
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, Data):
        raise TypeError(f"期望 torch_geometric.data.Data，得到 {type(data)}")

    for mask_name in ("train_mask", "val_mask", "test_mask"):
        if not hasattr(data, mask_name):
            raise AttributeError(f"Data 缺少 {mask_name}")
        m = getattr(data, mask_name)
        if m.dtype != torch.bool:
            setattr(data, mask_name, m.bool())

    if data.y.dim() > 1:
        data.y = data.y.view(-1)

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
        raise ValueError(f"dataset 应为 cora / pubmed / citeseer，got {name}")
    dataset = Planetoid(root=root, name=name_map[key])
    data = dataset[0]
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
    解析训练设备。
    - --device cpu：强制 CPU。
    - --device cuda 或 cuda:N：按字符串使用（显式 cuda:N 优先于 --gpu）。
    - --gpu K：使用 cuda:K（在未指定带索引的 cuda:N 时生效）。
    - 均未指定：有 CUDA 则用 cuda:0，否则 CPU。
    """
    if device is not None and device.lower() == "cpu":
        return torch.device("cpu")

    if device is not None and ":" in device and device.lower().startswith("cuda"):
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"指定了 {device}，但当前环境不可用 CUDA。")
        idx = dev.index if dev.index is not None else 0
        if torch.cuda.is_available() and (idx < 0 or idx >= torch.cuda.device_count()):
            raise RuntimeError(
                f"设备 {device} 无效，可见 GPU 数量为 {torch.cuda.device_count()}。"
            )
        return dev

    if gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("已设置 --gpu，但当前环境不可用 CUDA。")
        n = torch.cuda.device_count()
        if gpu < 0 or gpu >= n:
            raise RuntimeError(f"--gpu={gpu} 无效，可见 GPU 数量为 {n}。")
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
) -> Path:
    """在 result_base 下创建本次运行的子目录，写入 metrics.json 与 best_model.pt。"""
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
                },
                "hyperparameters": _args_to_jsonable(args),
                "data_source": info.source,
            },
            run_dir / "best_model.pt",
        )

    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PyG node classification (GCN / GraphSAGE / GAT / GATv2)"
    )
    p.add_argument(
        "--source",
        type=str,
        default="llaga",
        choices=("llaga", "planetoid"),
        help="llaga: 使用 LLaGA/dataset/*/processed_data.pt；planetoid: PyG 官方数据",
    )
    p.add_argument("--dataset", type=str, default="cora", help="cora | pubmed（planetoid 时可加 citeseer）")
    p.add_argument(
        "--llaga_dataset_root",
        type=str,
        default=None,
        help="LLaGA 的 dataset 目录；默认为本仓库 LLaGA/dataset",
    )
    p.add_argument("--data_root", type=str, default="./data", help="仅 planetoid：数据下载目录")
    p.add_argument("--model", type=str, default="gcn", help="gcn | sage | gat | gatv2")
    p.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="隐层维度（GAT/GATv2 为每头输出维）",
    )
    p.add_argument("--num_layers", type=int, default=2, help="消息传递层数")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument(
        "--heads", type=int, default=8, help="GAT/GATv2 注意力头数（仅 gat/gatv2）"
    )
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="训练 batch：0=全图（默认）；>0 时用 NeighborLoader（需 torch-sparse 或 pyg-lib），否则自动退回全图",
    )
    p.add_argument(
        "--neighbor_fanout",
        type=int,
        default=25,
        help="仅 mini-batch 时有效：每层邻居采样数，重复 num_layers 次",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA 设备编号（如 0、1），等价于使用 cuda:该编号；与 --device cuda:N 二选一即可",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备：cpu / cuda / cuda:N；默认自动选 GPU0 或 CPU。显式 cuda:N 时优先于 --gpu",
    )
    p.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="结果保存目录，默认为本目录下的 result/",
    )
    p.add_argument("--no_save", action="store_true", help="不保存 metrics 与 checkpoint")
    return p.parse_args()


def run_gnn(args: argparse.Namespace | None = None) -> float:
    if args is None:
        args = parse_args()

    set_seed(args.seed)
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

    use_mini_batch_requested = args.batch_size > 0
    need_cpu_copy = use_mini_batch_requested and _neighbor_sampling_backend_available()
    if need_cpu_copy:
        # Data.to() 会原地修改对象；NeighborLoader 用 CPU 图，须在 to(device) 前 clone
        data_cpu = info.data.clone()
    else:
        data_cpu = None
    data = info.data.to(device)

    model = build_model(
        name=args.model,
        in_dim=info.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=info.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = 0.0
    best_state = None
    test_acc_at_best = 0.0

    use_mini_batch = use_mini_batch_requested and _neighbor_sampling_backend_available()
    train_loader = None
    if use_mini_batch_requested and not use_mini_batch:
        print(
            "未安装 torch-sparse 或 pyg-lib，无法使用 NeighborLoader；"
            "已忽略 --batch_size，改用全图训练。可: pip install torch-sparse 或 pyg-lib"
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
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bs
                n_train += bs
            loss_item = total_loss / max(n_train, 1)
        else:
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            loss_item = loss.item()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)
            test_acc = accuracy(logits, data.y, data.test_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            test_acc_at_best = test_acc

        if epoch % 10 == 0 or epoch == 1:
            train_acc = accuracy(logits, data.y, data.train_mask)
            print(
                f"Epoch {epoch:03d} | loss {loss_item:.4f} | "
                f"train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f}"
            )

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
        )
        print(f"Saved results to: {run_dir}")

    return float(test_acc_at_best)


if __name__ == "__main__":
    run_gnn()
