"""Logit-based eval for link prediction (yes/no over two-node subgraphs).

Single forward pass; the answer position (a single ' yes' / ' no' token) is
masked, the model fills it in, and we score the two candidate token logits.

Reports both accuracy (argmax match) and ROC-AUC (softmax(yes vs no) vs.
ground-truth label).

Usage
-----
    CUDA_VISIBLE_DEVICES=3 python examples/tmdlm/eval_lp_logit.py \
        --exp cora_lp_smoke \
        --dataset_name cora \
        --lora_path .models/tmdlm-cora-lp/checkpoint-XXX \
        --use_topology_mask True
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

import dllm
from dllm.data.graph import load_lp_dataset, get_lp_yesno_token_ids
from dllm.pipelines.tmdlm.utils import GraphDataCollator

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class EvalLPLogitArgs:
    exp: str = field(metadata={"help": "Experiment name for logging"})
    model_name_or_path: str = field(default="GSAI-ML/LLaDA-8B-Instruct")
    dataset_name: str = field(default="cora")
    split: str = field(default="test")
    batch_size: int = field(default=8)
    max_seq_len: int = field(default=2048)
    max_neighbors_per_hop: int = field(default=10)
    max_hops: int = field(default=2)
    use_topology_mask: bool = field(default=False)
    log_file: str = field(default="experiments/experiment_log.jsonl")
    lora_path: str | None = field(default=None)
    position_id_type: str = field(default="sequential")
    seed: int = field(default=42)
    lp_neg_ratio: int = field(default=1)
    max_samples: int = field(default=0)


@torch.no_grad()
def evaluate_lp(
    model,
    tokenizer,
    test_dataset,
    yesno_token_ids,
    batch_size: int = 8,
    use_topology_mask: bool = False,
    position_id_type: str = "sequential",
):
    """Returns (accuracy, auc, per_label_acc dict, n_samples, all_probs, all_gt)."""
    device = next(model.parameters()).device
    no_id, yes_id = yesno_token_ids
    yesno_tensor = torch.tensor([no_id, yes_id], device=device)

    collator = GraphDataCollator(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        position_id_type=position_id_type,
    )
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    correct_per = {0: 0, 1: 0}
    total_per = {0: 0, 1: 0}
    yes_probs: list[float] = []
    gts: list[int] = []
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Evaluating LP")
    for batch in pbar:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        input_ids = batch["input_ids"]
        cls_labels = batch["cls_labels"]
        label_indices = batch["label_token_indices"]  # [b, 1]
        b, l = input_ids.shape

        masked_input = input_ids.clone()
        b_arange = torch.arange(b, device=device)
        attn1d = batch.get("attention_mask")
        if attn1d is not None:
            attn1d = attn1d.clone()
        for j in range(label_indices.shape[1]):
            masked_input[b_arange, label_indices[:, j]] = tokenizer.mask_token_id
            if attn1d is not None:
                attn1d[b_arange, label_indices[:, j]] = 1

        if use_topology_mask and "topology_mask" in batch:
            topo = batch["topology_mask"]
            additive = torch.zeros(b, l, l, device=device, dtype=model.dtype)
            additive = additive.masked_fill(topo == 0, float("-inf"))
            attn_mask = additive.unsqueeze(1)
        else:
            attn_mask = attn1d

        pos_ids = batch.get("position_ids")
        forward_kwargs = dict(input_ids=masked_input, attention_mask=attn_mask)
        if pos_ids is not None:
            forward_kwargs["position_ids"] = pos_ids

        logits = model(**forward_kwargs).logits  # [b, l, V]
        ans_logits = logits[
            torch.arange(b, device=device).unsqueeze(1),
            label_indices,
        ].squeeze(1)  # [b, V]
        restricted = ans_logits[:, yesno_tensor]  # [b, 2]
        probs = F.softmax(restricted, dim=-1)     # [b, 2]
        preds = restricted.argmax(dim=-1)         # 0=no, 1=yes

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

        pbar.set_postfix(
            acc=f"{100.0 * correct / max(total, 1):.2f}%",
            n=total,
        )

    accuracy = 100.0 * correct / max(total, 1)
    per_label_acc = {
        "no": 100.0 * correct_per[0] / max(total_per[0], 1),
        "yes": 100.0 * correct_per[1] / max(total_per[1], 1),
    }

    # AUC. If only one class is present, AUC is undefined; return None.
    auc = None
    if len(set(gts)) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(gts, yes_probs))
        except Exception as e:
            logger.warning("AUC computation failed: %s", e)
    return accuracy, auc, per_label_acc, total, yes_probs, gts


def main():
    parser = transformers.HfArgumentParser(EvalLPLogitArgs)
    (args,) = parser.parse_args_into_dataclasses()

    model_path = dllm.utils.resolve_with_base_env(
        args.model_name_or_path, "BASE_MODELS_DIR"
    )
    logger.info("Loading model from %s ...", model_path)

    model_args = dllm.utils.ModelArguments(model_name_or_path=model_path)
    model = dllm.utils.get_model(model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args)

    if args.lora_path:
        from peft import PeftModel
        logger.info("Loading LoRA adapter from %s ...", args.lora_path)
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    model.eval()
    logger.info("Model loaded on %s, dtype=%s", next(model.parameters()).device, model.dtype)

    yesno_ids = get_lp_yesno_token_ids(tokenizer)
    logger.info("yes/no token ids: no=%d, yes=%d", yesno_ids[0], yesno_ids[1])

    logger.info("Loading %s LP %s split ...", args.dataset_name, args.split)
    test_dataset = load_lp_dataset(
        args.dataset_name,
        tokenizer=tokenizer,
        split=args.split,
        max_seq_len=args.max_seq_len,
        max_neighbors_per_hop=args.max_neighbors_per_hop,
        max_hops=args.max_hops,
        seed=args.seed,
        neg_ratio=args.lp_neg_ratio,
        max_samples=args.max_samples,
    )
    logger.info("Loaded %d LP samples", len(test_dataset))

    sample = test_dataset[0]
    logger.info(
        "Sample 0: seq_len=%d cls_label=%d label_pos=%d roles=%s",
        len(sample["input_ids"]),
        sample["cls_label"],
        sample["label_token_pos"],
        sample.get("node_roles"),
    )
    logger.info(
        "Sample 0 text head: %s",
        tokenizer.decode(sample["input_ids"][:160]),
    )

    t_start = time.time()
    accuracy, auc, per_label_acc, n, _yes_probs, _gts = evaluate_lp(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        yesno_token_ids=yesno_ids,
        batch_size=args.batch_size,
        use_topology_mask=args.use_topology_mask,
        position_id_type=args.position_id_type,
    )
    elapsed = time.time() - t_start

    logger.info("=" * 60)
    logger.info("Experiment: %s", args.exp)
    logger.info("Samples: %d", n)
    logger.info("Accuracy: %.2f%%", accuracy)
    logger.info("AUC: %s", f"{auc:.4f}" if auc is not None else "n/a")
    logger.info(
        "Per-label accuracy: no=%.2f%% yes=%.2f%%",
        per_label_acc["no"], per_label_acc["yes"],
    )
    logger.info("Time: %.1f s", elapsed)

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": args.exp,
        "task": "lp",
        "model": args.model_name_or_path,
        "dataset": args.dataset_name,
        "split": args.split,
        "n_samples": n,
        "accuracy": round(accuracy, 2),
        "auc": round(auc, 4) if auc is not None else None,
        "per_label_accuracy": {k: round(v, 2) for k, v in per_label_acc.items()},
        "config": {
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "max_neighbors_per_hop": args.max_neighbors_per_hop,
            "max_hops": args.max_hops,
            "use_topology_mask": args.use_topology_mask,
            "lora_path": args.lora_path,
            "lp_neg_ratio": args.lp_neg_ratio,
            "seed": args.seed,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(args.log_file, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    logger.info("Results logged to %s", args.log_file)


if __name__ == "__main__":
    main()
