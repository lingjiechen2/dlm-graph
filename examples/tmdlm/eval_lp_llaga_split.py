"""Eval LP on LLaGA's exact test split for head-to-head comparison.

Loads test node-pairs from LLaGA's ``edge_sampled_2_10_only_test.jsonl``,
builds adj_train from ``processed_data_link_notest.pt`` (test edges already
removed), then runs the same logit-scoring eval as ``eval_lp_logit.py``.

This gives numbers directly comparable to LLaGA Table 1 (same test pairs,
same pos/neg ratio).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_lp_llaga_split.py \
        --exp cora_lp_llaga_split \
        --dataset_name cora \
        --lora_path .models/<run>/checkpoint-XXX \
        --max_seq_len 4096
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

import dllm
from dllm.data.graph import (
    build_edge_sample,
    get_lp_yesno_token_ids,
    _sample_lp_neighbors,
    _sample_lp_neighbors_with_edges,
    _truncate_text_for_neighbor,
)
from dllm.data.datasets._common import build_adjacency, load_llaga_processed_data
from dllm.pipelines.tmdlm.utils import GraphDataCollator

logger = dllm.utils.get_default_logger(__name__)

_LLAGA_DATA_ROOT = Path(
    os.environ.get(
        "LLAGA_DATA_ROOT",
        "/home/lingjie7/auto-research/projects/dlm-graph/.datasets/llaga",
    )
)


@dataclass
class EvalArgs:
    exp: str = field(metadata={"help": "Experiment name for logging"})
    dataset_name: str = field(default="cora")
    model_name_or_path: str = field(default="GSAI-ML/LLaDA-8B-Instruct")
    lora_path: str | None = field(default=None)
    batch_size: int = field(default=8)
    max_seq_len: int = field(default=4096)
    max_neighbors_per_hop: int = field(default=10)
    max_hops: int = field(default=2)
    use_topology_mask: bool = field(default=False)
    topology_mask_type: str = field(default="star")
    position_id_type: str = field(default="sequential")
    seed: int = field(default=42)
    log_file: str = field(default=".models/eval_logs/lp_llaga_split.jsonl")
    max_samples: int = field(default=0, metadata={"help": "0 = use all"})


def _load_llaga_test_pairs(dataset_name: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Read LLaGA's test JSONL and return (pos_edges, neg_edges)."""
    jsonl = _LLAGA_DATA_ROOT / dataset_name / "edge_sampled_2_10_only_test.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"LLaGA test split not found: {jsonl}")
    pos, neg = [], []
    with open(jsonl) as f:
        for line in f:
            obj = json.loads(line)
            u, v = int(obj["id"][0]), int(obj["id"][1])
            label_str = obj["conversations"][1]["value"]  # "yes" or "no"
            if label_str == "yes":
                pos.append((u, v))
            else:
                neg.append((u, v))
    return pos, neg


def _load_adj_notest(dataset_name: str) -> dict[int, list[int]]:
    """Build adj_train from LLaGA's processed_data_link_notest.pt."""
    pt = _LLAGA_DATA_ROOT / dataset_name / "processed_data_link_notest.pt"
    if not pt.exists():
        raise FileNotFoundError(f"LLaGA notest graph not found: {pt}")
    data = torch.load(pt, map_location="cpu", weights_only=False)
    edge_index = data.edge_index.cpu().numpy()
    return build_adjacency(edge_index)


def _load_node_data(dataset_name: str) -> dict[int, dict]:
    """Load node text (title + abstract) via our dataset loaders.

    Passes llaga_dir so loaders can use the local LLaGA processed_data.pt
    cache instead of re-downloading from HuggingFace.
    """
    from dllm.data.datasets import cora, ogbn_arxiv, pubmed
    _loaders = {"cora": cora, "ogbn-arxiv": ogbn_arxiv, "pubmed": pubmed}
    if dataset_name not in _loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    config = {"llaga_dir": str(_LLAGA_DATA_ROOT / dataset_name)}
    node_data, _adj, _cls, _ids = _loaders[dataset_name].load(config, "train", seed=42)
    return node_data


def _build_samples(
    pos_edges: list[tuple[int, int]],
    neg_edges: list[tuple[int, int]],
    node_data: dict[int, dict],
    adj_train: dict[int, list[int]],
    tokenizer,
    max_seq_len: int,
    max_neighbors_per_hop: int,
    max_hops: int,
    seed: int,
    max_samples: int,
    include_topology_edges: bool = False,
    shard_rank: int = 0,
    shard_world: int = 1,
) -> list[dict]:
    rng = random.Random(seed)
    pairs = [((u, v), 1) for u, v in pos_edges] + [((u, v), 0) for u, v in neg_edges]
    rng.shuffle(pairs)
    if max_samples:
        pairs = pairs[:max_samples]
    # DDP eval: each rank evaluates 1/world_size of the pairs (strided after
    # the seeded shuffle so partitions are balanced in yes/no ratio).
    if shard_world > 1:
        pairs = pairs[shard_rank::shard_world]

    samples = []
    for (u, v), label in pairs:
        if u not in node_data or v not in node_data:
            continue
        if include_topology_edges:
            (
                u_nb_ids,
                u_nb_hops,
                u_nb_edges,
                v_nb_ids,
                v_nb_hops,
                v_nb_edges,
            ) = _sample_lp_neighbors_with_edges(
                adj_train, u, v, max_neighbors_per_hop, max_hops, rng
            )
        else:
            u_nb_ids, u_nb_hops, v_nb_ids, v_nb_hops = _sample_lp_neighbors(
                adj_train, u, v, max_neighbors_per_hop, max_hops, rng
            )
            u_nb_edges = None
            v_nb_edges = None
        sample = build_edge_sample(
            u_text=_truncate_text_for_neighbor(node_data, u),
            v_text=_truncate_text_for_neighbor(node_data, v),
            u_neighbor_texts=[_truncate_text_for_neighbor(node_data, nb) for nb in u_nb_ids],
            u_neighbor_hops=u_nb_hops,
            v_neighbor_texts=[_truncate_text_for_neighbor(node_data, nb) for nb in v_nb_ids],
            v_neighbor_hops=v_nb_hops,
            cls_label=label,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            u_neighbor_edges=u_nb_edges,
            v_neighbor_edges=v_nb_edges,
        )
        sample["edge"] = (int(u), int(v))
        samples.append(sample)
    return samples


@torch.no_grad()
def evaluate(model, tokenizer, dataset, yesno_ids, args: EvalArgs):
    device = next(model.parameters()).device
    no_id, yes_id = yesno_ids
    yesno_tensor = torch.tensor([no_id, yes_id], device=device)

    collator = GraphDataCollator(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        position_id_type=args.position_id_type,
        topology_mask_type=args.topology_mask_type,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    correct = total = 0
    correct_per = {0: 0, 1: 0}
    total_per = {0: 0, 1: 0}
    yes_probs: list[float] = []
    gts: list[int] = []

    pbar = tqdm(loader, desc="Evaluating")
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        input_ids = batch["input_ids"]
        cls_labels = batch["cls_labels"]
        label_indices = batch["label_token_indices"]
        b, l = input_ids.shape

        masked = input_ids.clone()
        b_idx = torch.arange(b, device=device)
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = attn.clone()
        for j in range(label_indices.shape[1]):
            masked[b_idx, label_indices[:, j]] = tokenizer.mask_token_id
            if attn is not None:
                attn[b_idx, label_indices[:, j]] = 1

        if args.use_topology_mask and "topology_mask" in batch:
            topo = batch["topology_mask"]
            additive = torch.zeros(b, l, l, device=device, dtype=model.dtype)
            additive = additive.masked_fill(topo == 0, float("-inf"))
            attn_mask = additive.unsqueeze(1)
        else:
            attn_mask = attn

        fwd = dict(input_ids=masked, attention_mask=attn_mask)
        pos_ids = batch.get("position_ids")
        if pos_ids is not None:
            fwd["position_ids"] = pos_ids

        logits = model(**fwd).logits
        ans_logits = logits[b_idx.unsqueeze(1), label_indices].squeeze(1)
        restricted = ans_logits[:, yesno_tensor]
        probs = F.softmax(restricted, dim=-1)
        preds = restricted.argmax(dim=-1)

        for i in range(b):
            gt = int(cls_labels[i].item())
            pred = int(preds[i].item())
            total += 1
            total_per[gt] += 1
            if pred == gt:
                correct += 1
                correct_per[gt] += 1
            yes_probs.append(float(probs[i, 1].item()))
            gts.append(gt)

        pbar.set_postfix(acc=f"{100.0 * correct / max(total, 1):.2f}%", n=total)

    # Return raw local counts; aggregation across DDP ranks happens in main().
    return {
        "correct": correct,
        "total": total,
        "correct_per": correct_per,
        "total_per": total_per,
        "yes_probs": yes_probs,
        "gts": gts,
    }


def main():
    # DDP setup. Each rank shards the test pairs (1/world_size each), runs
    # inference independently, then all_reduce + all_gather aggregates counts
    # and yes_probs for the global accuracy/AUC. Single-GPU usage is unchanged
    # — when WORLD_SIZE == 1, dist is not initialized and shard math is no-op.
    from datetime import timedelta
    import torch.distributed as dist
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=int(os.environ.get("DDP_INIT_TIMEOUT", "3600"))),
        )
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    parser = transformers.HfArgumentParser(EvalArgs)
    (args,) = parser.parse_args_into_dataclasses()

    model_path = dllm.utils.resolve_with_base_env(args.model_name_or_path, "BASE_MODELS_DIR")
    model_args = dllm.utils.ModelArguments(model_name_or_path=model_path)
    model = dllm.utils.get_model(model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args)

    if args.lora_path:
        from peft import PeftModel
        logger.info("Loading LoRA from %s", args.lora_path)
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    model.eval()
    logger.info("Model on %s dtype=%s", next(model.parameters()).device, model.dtype)

    yesno_ids = get_lp_yesno_token_ids(tokenizer)

    logger.info("Loading LLaGA test split for %s ...", args.dataset_name)
    pos_edges, neg_edges = _load_llaga_test_pairs(args.dataset_name)
    logger.info("LLaGA test: %d pos + %d neg = %d total", len(pos_edges), len(neg_edges), len(pos_edges) + len(neg_edges))

    logger.info("Loading adj_train (notest) ...")
    adj_train = _load_adj_notest(args.dataset_name)

    logger.info("Loading node text data ...")
    node_data = _load_node_data(args.dataset_name)

    logger.info("[rank %d/%d] Building text prompts (shard) ...", rank, world_size)
    dataset = _build_samples(
        pos_edges, neg_edges, node_data, adj_train, tokenizer,
        args.max_seq_len, args.max_neighbors_per_hop, args.max_hops,
        args.seed, args.max_samples,
        include_topology_edges=(args.topology_mask_type == "khop_tree"),
        shard_rank=rank, shard_world=world_size,
    )
    logger.info("[rank %d/%d] Built %d local samples", rank, world_size, len(dataset))

    t0 = time.time()
    out = evaluate(model, tokenizer, dataset, yesno_ids, args)
    elapsed = time.time() - t0

    # Aggregate across ranks
    correct = out["correct"]; total = out["total"]
    cper, tper = out["correct_per"], out["total_per"]
    yes_probs = out["yes_probs"]; gts = out["gts"]
    if world_size > 1:
        device = next(model.parameters()).device
        counts = torch.tensor(
            [correct, total, cper[0], tper[0], cper[1], tper[1]],
            device=device, dtype=torch.long,
        )
        dist.all_reduce(counts)
        c_correct, c_total, c0, t0c, c1, t1c = counts.tolist()
        correct, total = c_correct, c_total
        cper = {0: c0, 1: c1}; tper = {0: t0c, 1: t1c}
        # AUC needs concatenated probs/gts on rank 0
        gathered_probs = [None] * world_size
        gathered_gts = [None] * world_size
        dist.all_gather_object(gathered_probs, yes_probs)
        dist.all_gather_object(gathered_gts, gts)
        yes_probs = sum(gathered_probs, [])
        gts = sum(gathered_gts, [])

    accuracy = 100.0 * correct / max(total, 1)
    per_label = {
        "no": 100.0 * cper[0] / max(tper[0], 1),
        "yes": 100.0 * cper[1] / max(tper[1], 1),
    }
    auc = None
    if len(set(gts)) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(gts, yes_probs))
        except Exception as e:
            logger.warning("AUC failed: %s", e)

    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return

    logger.info("=" * 60)
    logger.info("exp=%s  dataset=%s  n=%d  world_size=%d", args.exp, args.dataset_name, total, world_size)
    logger.info("Accuracy:  %.2f%%", accuracy)
    logger.info("AUC:       %s", f"{auc:.4f}" if auc else "n/a")
    logger.info("Per-label: no=%.2f%%  yes=%.2f%%", per_label["no"], per_label["yes"])
    logger.info("Time:      %.1fs", elapsed)

    os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "exp": args.exp,
        "dataset": args.dataset_name,
        "split": "llaga_test",
        "n_samples": total,
        "accuracy": round(accuracy, 2),
        "auc": round(auc, 4) if auc else None,
        "per_label_accuracy": {k: round(v, 2) for k, v in per_label.items()},
        "config": {
            "lora_path": args.lora_path,
            "max_seq_len": args.max_seq_len,
            "max_neighbors_per_hop": args.max_neighbors_per_hop,
            "max_hops": args.max_hops,
            "use_topology_mask": args.use_topology_mask,
            "topology_mask_type": args.topology_mask_type,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "world_size": world_size,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(args.log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.destroy_process_group()
    logger.info("Logged to %s", args.log_file)


if __name__ == "__main__":
    main()
