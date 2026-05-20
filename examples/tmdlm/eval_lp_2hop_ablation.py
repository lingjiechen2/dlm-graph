"""Ablation: does 2-hop common-neighbor signal inflate LP performance?

Evaluates the same checkpoint under two conditions on LLaGA's test split:
  filtered   -- current behaviour: v removed from ALL hops of u's subgraph
  unfiltered -- ablation: v can appear in u's 2-hop via common neighbours
               (direct edge already absent because notest adj is used)

For each condition the script also stratifies results into:
  Group A (leaking) -- v appears in u's 2-hop when unfiltered (131/301 Cora pos)
  Group B (clean)   -- v does not appear in u's 2-hop when unfiltered

Usage
-----
    CUDA_VISIBLE_DEVICES=2 python examples/tmdlm/eval_lp_2hop_ablation.py \
        --lora_path .models/<run>/checkpoint-XXX \
        --dataset_name cora \
        --max_seq_len 4096 \
        --use_topology_mask True \
        --log_file .models/eval_logs/lp_2hop_ablation.jsonl
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
    _sample_khop_neighbors,
    _truncate_text_for_neighbor,
    build_edge_sample,
    get_lp_yesno_token_ids,
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
    dataset_name: str = field(default="cora")
    model_name_or_path: str = field(default="GSAI-ML/LLaDA-8B-Instruct")
    lora_path: str | None = field(default=None)
    batch_size: int = field(default=8)
    max_seq_len: int = field(default=4096)
    max_neighbors_per_hop: int = field(default=10)
    max_hops: int = field(default=2)
    use_topology_mask: bool = field(default=False)
    position_id_type: str = field(default="sequential")
    seed: int = field(default=42)
    log_file: str = field(default=".models/eval_logs/lp_2hop_ablation.jsonl")
    max_samples: int = field(default=0)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_llaga_test_pairs(dataset_name: str):
    jsonl = _LLAGA_DATA_ROOT / dataset_name / "edge_sampled_2_10_only_test.jsonl"
    pos, neg = [], []
    with open(jsonl) as f:
        for line in f:
            obj = json.loads(line)
            u, v = int(obj["id"][0]), int(obj["id"][1])
            if obj["conversations"][1]["value"] == "yes":
                pos.append((u, v))
            else:
                neg.append((u, v))
    return pos, neg


def _load_adj_notest(dataset_name: str) -> dict[int, list[int]]:
    pt = _LLAGA_DATA_ROOT / dataset_name / "processed_data_link_notest.pt"
    data = torch.load(pt, map_location="cpu", weights_only=False)
    return build_adjacency(data.edge_index.cpu().numpy())


def _load_node_data(dataset_name: str) -> dict[int, dict]:
    from dllm.data.datasets import cora, ogbn_arxiv, pubmed
    _loaders = {"cora": cora, "ogbn-arxiv": ogbn_arxiv, "pubmed": pubmed}
    config = {"llaga_dir": str(_LLAGA_DATA_ROOT / dataset_name)}
    node_data, _adj, _cls, _ids = _loaders[dataset_name].load(config, "train", seed=42)
    return node_data


def _sample_lp_neighbors_filtered(adj, u, v, max_nb, max_hops, rng):
    """Current behaviour: remove v from u's pool (all hops) and u from v's pool."""
    u_ids, u_hops = _sample_khop_neighbors(adj, u, max_nb, max_hops, rng)
    u_kept = [(nb, h) for nb, h in zip(u_ids, u_hops) if nb != v]
    v_ids, v_hops = _sample_khop_neighbors(adj, v, max_nb, max_hops, rng)
    v_kept = [(nb, h) for nb, h in zip(v_ids, v_hops) if nb != u]
    return (
        [nb for nb, _ in u_kept], [h for _, h in u_kept],
        [nb for nb, _ in v_kept], [h for _, h in v_kept],
    )


def _sample_lp_neighbors_unfiltered(adj, u, v, max_nb, max_hops, rng):
    """Ablation: do NOT remove v from u's pool (direct edge already absent in notest adj).
    v can still appear in u's 2-hop via shared neighbours."""
    u_ids, u_hops = _sample_khop_neighbors(adj, u, max_nb, max_hops, rng)
    v_ids, v_hops = _sample_khop_neighbors(adj, v, max_nb, max_hops, rng)
    return u_ids, u_hops, v_ids, v_hops


def _sample_lp_neighbors_extreme(adj, u, v, max_nb, max_hops, rng, label: int):
    """Extreme ablation: for positive pairs forcefully inject v into u's 2-hop list
    (and u into v's 2-hop list) if not already present. Negative pairs are
    left unfiltered (v won't appear naturally since there is no edge)."""
    u_ids, u_hops = _sample_khop_neighbors(adj, u, max_nb, max_hops, rng)
    v_ids, v_hops = _sample_khop_neighbors(adj, v, max_nb, max_hops, rng)
    if label == 1:
        if v not in u_ids:
            u_ids = u_ids + [v]
            u_hops = u_hops + [2]
        if u not in v_ids:
            v_ids = v_ids + [u]
            v_hops = v_hops + [2]
    return u_ids, u_hops, v_ids, v_hops


def _v_in_2hop(adj, u, v, max_nb, max_hops, rng) -> bool:
    """Would v appear in u's subgraph without the nb!=v filter?"""
    rng2 = random.Random(rng.random())  # snapshot so main rng is unaffected
    u_ids, _ = _sample_khop_neighbors(adj, u, max_nb, max_hops, rng2)
    return v in set(u_ids)


# condition name -> (filter_candidate, extreme)
_CONDITIONS = [
    ("filtered",   True,  False),   # current behaviour
    ("unfiltered", False, False),   # natural 2-hop leakage (~43% pos)
    ("extreme",    False, True),    # force v into 2-hop for ALL pos pairs
]


def _build_samples(pos_edges, neg_edges, node_data, adj_train,
                   tokenizer, args: EvalArgs,
                   filter_candidate: bool, extreme: bool = False):
    rng = random.Random(args.seed)
    pairs = [((u, v), 1) for u, v in pos_edges] + [((u, v), 0) for u, v in neg_edges]
    rng.shuffle(pairs)
    if args.max_samples:
        pairs = pairs[:args.max_samples]

    # Pre-compute leaking group (fixed, condition-independent)
    group_rng = random.Random(args.seed)
    leak_set: set[tuple[int, int]] = set()
    for (u, v), label in pairs:
        if label == 1 and _v_in_2hop(adj_train, u, v, args.max_neighbors_per_hop, args.max_hops, group_rng):
            leak_set.add((u, v))

    rng2 = random.Random(args.seed)
    samples = []
    for (u, v), label in pairs:
        if u not in node_data or v not in node_data:
            continue
        if extreme:
            u_nb_ids, u_nb_hops, v_nb_ids, v_nb_hops = _sample_lp_neighbors_extreme(
                adj_train, u, v, args.max_neighbors_per_hop, args.max_hops, rng2, label
            )
        elif filter_candidate:
            u_nb_ids, u_nb_hops, v_nb_ids, v_nb_hops = _sample_lp_neighbors_filtered(
                adj_train, u, v, args.max_neighbors_per_hop, args.max_hops, rng2
            )
        else:
            u_nb_ids, u_nb_hops, v_nb_ids, v_nb_hops = _sample_lp_neighbors_unfiltered(
                adj_train, u, v, args.max_neighbors_per_hop, args.max_hops, rng2
            )
        sample = build_edge_sample(
            u_text=_truncate_text_for_neighbor(node_data, u),
            v_text=_truncate_text_for_neighbor(node_data, v),
            u_neighbor_texts=[_truncate_text_for_neighbor(node_data, nb) for nb in u_nb_ids],
            u_neighbor_hops=u_nb_hops,
            v_neighbor_texts=[_truncate_text_for_neighbor(node_data, nb) for nb in v_nb_ids],
            v_neighbor_hops=v_nb_hops,
            cls_label=label,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
        )
        sample["edge"] = (int(u), int(v))
        # Group A: v naturally appears in 2-hop (common neighbours)
        # Group B: v does not appear in 2-hop naturally
        # In extreme condition Group A is expanded to ALL positive pairs
        if extreme and label == 1:
            sample["group"] = "A_leak"
        else:
            sample["group"] = "A_leak" if (u, v) in leak_set else "B_clean"
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, dataset, yesno_ids, args: EvalArgs):
    device = next(model.parameters()).device
    no_id, yes_id = yesno_ids
    yesno_tensor = torch.tensor([no_id, yes_id], device=device)

    collator = GraphDataCollator(
        tokenizer=tokenizer, padding=True, return_tensors="pt",
        position_id_type=args.position_id_type,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    results = []  # per-sample: {edge, label, pred, yes_prob, group}
    pbar = tqdm(loader, desc="Evaluating")
    sample_idx = 0

    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        input_ids = batch["input_ids"]
        cls_labels = batch["cls_labels"]
        label_indices = batch["label_token_indices"]
        b, l = input_ids.shape
        b_idx = torch.arange(b, device=device)

        masked = input_ids.clone()
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
        if (pos_ids := batch.get("position_ids")) is not None:
            fwd["position_ids"] = pos_ids

        logits = model(**fwd).logits
        ans_logits = logits[b_idx.unsqueeze(1), label_indices].squeeze(1)
        restricted = ans_logits[:, yesno_tensor]
        probs = F.softmax(restricted, dim=-1)
        preds = restricted.argmax(dim=-1)

        for i in range(b):
            results.append({
                "edge": dataset[sample_idx]["edge"],
                "group": dataset[sample_idx]["group"],
                "label": int(cls_labels[i].item()),
                "pred": int(preds[i].item()),
                "yes_prob": float(probs[i, 1].item()),
            })
            sample_idx += 1

        acc_so_far = sum(r["pred"] == r["label"] for r in results) / len(results)
        pbar.set_postfix(acc=f"{100*acc_so_far:.2f}%")

    return results


def _summarise(results, label: str):
    from sklearn.metrics import roc_auc_score

    def _metrics(subset):
        if not subset:
            return None
        acc = 100 * sum(r["pred"] == r["label"] for r in subset) / len(subset)
        gts = [r["label"] for r in subset]
        probs = [r["yes_prob"] for r in subset]
        try:
            auc = roc_auc_score(gts, probs) if len(set(gts)) == 2 else None
        except Exception:
            auc = None
        pos_acc = 100 * sum(r["pred"] == 1 for r in subset if r["label"] == 1) / max(1, sum(r["label"] == 1 for r in subset))
        neg_acc = 100 * sum(r["pred"] == 0 for r in subset if r["label"] == 0) / max(1, sum(r["label"] == 0 for r in subset))
        return {"n": len(subset), "acc": round(acc, 2), "auc": round(auc, 4) if auc else None,
                "yes_acc": round(pos_acc, 2), "no_acc": round(neg_acc, 2)}

    all_m  = _metrics(results)
    A_m    = _metrics([r for r in results if r["group"] == "A_leak"])
    B_m    = _metrics([r for r in results if r["group"] == "B_clean"])

    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"  Overall   n={all_m['n']:4d}  acc={all_m['acc']:.2f}%  auc={all_m['auc']}")
    print(f"  Group A (leaking)  n={A_m['n'] if A_m else 0}  acc={A_m['acc'] if A_m else 'n/a'}%  auc={A_m['auc'] if A_m else 'n/a'}")
    print(f"  Group B (clean)    n={B_m['n'] if B_m else 0}  acc={B_m['acc'] if B_m else 'n/a'}%  auc={B_m['auc'] if B_m else 'n/a'}")
    print(f"{'='*60}\n")
    return {"condition": label, "overall": all_m, "group_A_leak": A_m, "group_B_clean": B_m}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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
    yesno_ids = get_lp_yesno_token_ids(tokenizer)

    pos_edges, neg_edges = _load_llaga_test_pairs(args.dataset_name)
    adj_train = _load_adj_notest(args.dataset_name)
    node_data  = _load_node_data(args.dataset_name)

    logger.info("LLaGA test: %d pos + %d neg", len(pos_edges), len(neg_edges))

    all_results = {}
    for cond_name, filter_candidate, extreme in _CONDITIONS:
        logger.info("Building samples: condition=%s", cond_name)
        dataset = _build_samples(
            pos_edges, neg_edges, node_data, adj_train,
            tokenizer, args, filter_candidate=filter_candidate, extreme=extreme,
        )
        logger.info("Evaluating %d samples ...", len(dataset))
        t0 = time.time()
        results = evaluate(model, tokenizer, dataset, yesno_ids, args)
        elapsed = time.time() - t0
        summary = _summarise(results, cond_name)
        summary["elapsed_seconds"] = round(elapsed, 1)
        all_results[cond_name] = summary

    # Log
    os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset_name,
        "lora_path": args.lora_path,
        "max_seq_len": args.max_seq_len,
        "use_topology_mask": args.use_topology_mask,
        "results": all_results,
    }
    with open(args.log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Logged to %s", args.log_file)


if __name__ == "__main__":
    main()
