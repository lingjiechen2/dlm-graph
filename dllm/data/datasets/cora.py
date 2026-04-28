"""Cora loader (HuggingFace TAPE + PyG Planetoid, with optional LLaGA cache)."""

from __future__ import annotations

from datasets import load_dataset

from ._common import build_adjacency, load_llaga_processed_data


def load(
    config: dict,
    split: str,
    seed: int,
) -> tuple[dict, dict, list[str], list[int]]:
    llaga = load_llaga_processed_data(config, split)
    if llaga is not None:
        return llaga

    hf_split = {"train": "train", "val": "validation", "test": "test"}[split]
    tape_ds = load_dataset(config["hf_path"], cache_dir=str(config["cache_dir"]))

    node_data: dict[int, dict] = {}
    label_to_class: dict[int, str] = {}
    for s in tape_ds:
        for sample in tape_ds[s]:
            node_data[sample["id"]] = {
                "title": sample["T"],
                "abstract": sample["A"],
                "label": sample["label"],
            }
            label_to_class[sample["label"]] = sample["class"]
    class_names = [label_to_class[i] for i in range(len(label_to_class))]

    from torch_geometric.datasets import Planetoid

    pyg_ds = Planetoid(root=str(config["pyg_root"]), name=config["pyg_name"])
    adj = build_adjacency(pyg_ds[0].edge_index.numpy())

    split_ids = [sample["id"] for sample in tape_ds[hf_split]]
    return node_data, adj, class_names, split_ids
