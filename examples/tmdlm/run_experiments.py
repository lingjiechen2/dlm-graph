"""
TM-DLM Experiment Runner: Layer 0 and SFT evaluation on TAG datasets.

Usage
-----
Layer 0 (train-free, target-only, no neighbors):
    CUDA_VISIBLE_DEVICES=2 python examples/tmdlm/run_experiments.py \
        --exp baseline_no_neighbors \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --dataset_name cora --max_hops 0

Layer 0 (train-free, with neighbors, no topology mask):
    CUDA_VISIBLE_DEVICES=2 python examples/tmdlm/run_experiments.py \
        --exp neighbors_full_attn \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --dataset_name cora

Layer 0 (train-free, with neighbors + topology mask):
    CUDA_VISIBLE_DEVICES=2 python examples/tmdlm/run_experiments.py \
        --exp tmdlm_layer0 \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --dataset_name cora --use_topology_mask True
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

import dllm
from dllm.data.graph import load_tag_dataset, get_class_token_ids, DATASET_CONFIGS
from dllm.pipelines.tmdlm.utils import GraphDataCollator

logger = dllm.utils.get_default_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TM-DLM Experiment Runner")
    parser.add_argument(
        "--exp", type=str, required=True, help="Experiment name for logging"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct"
    )
    parser.add_argument("--dataset_name", type=str, default="cora")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_neighbors_per_hop", type=int, default=10)
    parser.add_argument("--max_hops", type=int, default=2)
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=1,
        help="Number of denoising steps (1 = single forward pass)",
    )
    parser.add_argument(
        "--use_topology_mask",
        type=bool,
        default=False,
        help="Apply topology mask to attention",
    )
    parser.add_argument(
        "--log_file", type=str, default="experiments/experiment_log.jsonl"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint (for evaluating fine-tuned models)",
    )
    parser.add_argument(
        "--position_id_type",
        type=str,
        default="sequential",
        choices=["sequential", "topological"],
        help="Position ID scheme: sequential (default) or topological (per-node reset)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def evaluate_layer0(
    model,
    tokenizer,
    test_dataset,
    class_first_token_ids,
    batch_size=8,
    use_topology_mask=False,
    denoising_steps=1,
    position_id_type="sequential",
):
    """
    Layer 0 evaluation: frozen model, mask ALL class-name tokens, forward pass,
    restricted argmax over class first-token IDs at the label position.

    Returns:
        accuracy: float
        per_class_correct: dict[int, int]
        per_class_total: dict[int, int]
        predictions: list[int]
    """
    device = next(model.parameters()).device
    num_classes = len(class_first_token_ids)
    class_token_ids_tensor = torch.tensor(class_first_token_ids, device=device)

    collator = GraphDataCollator(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        position_id_type=position_id_type,
    )
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    correct = 0
    total = 0
    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}
    all_predictions = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        input_ids = batch["input_ids"]  # [b, l]
        labels = batch["labels"]  # [b, l]
        cls_labels = batch["cls_labels"]  # [b]
        label_indices = batch["label_token_indices"]  # [b, 1]
        b, l = input_ids.shape

        # Mask ALL class-name tokens (positions where labels != -100)
        masked_input = input_ids.clone()
        maskable = labels != -100  # [b, l]
        masked_input[maskable] = tokenizer.mask_token_id

        # Build attention mask
        if use_topology_mask and "topology_mask" in batch:
            topo = batch["topology_mask"]  # [b, l, l]
            additive = torch.zeros(b, l, l, device=device, dtype=model.dtype)
            additive = additive.masked_fill(topo == 0, float("-inf"))
            attn_mask = additive.unsqueeze(1)  # [b, 1, l, l]
        else:
            # Standard padding-only mask (full attention among valid tokens)
            attn_mask = batch.get("attention_mask", None)  # [b, l]

        # Iterative denoising
        pos_ids = batch.get("position_ids", None)
        x = masked_input
        for step in range(denoising_steps):
            forward_kwargs = dict(input_ids=x, attention_mask=attn_mask)
            if pos_ids is not None:
                forward_kwargs["position_ids"] = pos_ids
            outputs = model(**forward_kwargs)
            logits = outputs.logits  # [b, l, V]

            if step < denoising_steps - 1:
                # Intermediate step: unmask with argmax predictions at masked positions
                pred_tokens = logits.argmax(dim=-1)  # [b, l]
                still_masked = x == tokenizer.mask_token_id
                x = torch.where(still_masked, pred_tokens, x)

        # Extract logits at the first label-token position
        label_logits = logits[
            torch.arange(b, device=device).unsqueeze(1),
            label_indices,
        ].squeeze(
            1
        )  # [b, V]

        # Restricted argmax: only consider class-name first tokens
        restricted_logits = label_logits[:, class_token_ids_tensor]  # [b, num_classes]
        preds = restricted_logits.argmax(dim=-1)  # [b] class indices

        all_predictions.extend(preds.cpu().tolist())

        for i in range(b):
            gt = cls_labels[i].item()
            pred = preds[i].item()
            per_class_total[gt] += 1
            if pred == gt:
                correct += 1
                per_class_correct[gt] += 1
            total += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, per_class_correct, per_class_total, all_predictions


def main():
    args = parse_args()

    # Resolve model path
    model_path = dllm.utils.resolve_with_base_env(
        args.model_name_or_path, "BASE_MODELS_DIR"
    )
    logger.info("Loading model from %s ...", model_path)

    # Load model and tokenizer
    model_args = dllm.utils.ModelArguments(model_name_or_path=model_path)
    model = dllm.utils.get_model(model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args)

    # Load LoRA adapter if specified
    if args.lora_path:
        from peft import PeftModel

        logger.info("Loading LoRA adapter from %s ...", args.lora_path)
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        logger.info("LoRA adapter merged.")

    model.eval()

    device = next(model.parameters()).device
    logger.info("Model loaded on %s, dtype=%s", device, model.dtype)

    # Get class token IDs
    class_names, class_first_token_ids = get_class_token_ids(
        args.dataset_name, tokenizer
    )
    logger.info("Class names: %s", class_names)
    logger.info("Class first token IDs: %s", class_first_token_ids)

    # Verify token IDs are unique
    if len(set(class_first_token_ids)) != len(class_first_token_ids):
        logger.warning(
            "WARNING: Some classes share the same first token ID! "
            "Classification will be ambiguous."
        )
        for i, (name, tid) in enumerate(zip(class_names, class_first_token_ids)):
            logger.info(
                "  Class %d: '%s' -> token %d ('%s')",
                i,
                name,
                tid,
                tokenizer.decode([tid]),
            )

    # Load test dataset
    logger.info("Loading %s %s split...", args.dataset_name, args.split)
    test_dataset = load_tag_dataset(
        args.dataset_name,
        tokenizer=tokenizer,
        split=args.split,
        max_seq_len=args.max_seq_len,
        max_neighbors_per_hop=args.max_neighbors_per_hop,
        max_hops=args.max_hops,
        seed=args.seed,
    )
    logger.info("Loaded %d samples", len(test_dataset))

    # Print a sample for sanity check
    sample = test_dataset[0]
    logger.info(
        "Sample 0: seq_len=%d, cls_label=%d, label_pos=%d",
        len(sample["input_ids"]),
        sample["cls_label"],
        sample["label_token_pos"],
    )
    logger.info(
        "Sample 0 text (first 200 chars): %s",
        tokenizer.decode(sample["input_ids"][:200]),
    )

    # Run evaluation
    t_start = time.time()
    accuracy, per_class_correct, per_class_total, predictions = evaluate_layer0(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        class_first_token_ids=class_first_token_ids,
        batch_size=args.batch_size,
        use_topology_mask=args.use_topology_mask,
        denoising_steps=args.denoising_steps,
        position_id_type=args.position_id_type,
    )
    elapsed = time.time() - t_start

    # Log results
    logger.info("=" * 60)
    logger.info("Experiment: %s", args.exp)
    logger.info(
        "Accuracy: %.2f%% (%d/%d)",
        accuracy,
        sum(per_class_correct.values()),
        sum(per_class_total.values()),
    )
    logger.info("Time: %.1f seconds", elapsed)

    for i, name in enumerate(class_names):
        c, t = per_class_correct[i], per_class_total[i]
        logger.info(
            "  Class %d (%s): %.1f%% (%d/%d)",
            i,
            name,
            100.0 * c / t if t > 0 else 0,
            c,
            t,
        )

    # Save to log file
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": args.exp,
        "model": args.model_name_or_path,
        "dataset": args.dataset_name,
        "split": args.split,
        "accuracy": round(accuracy, 2),
        "per_class_accuracy": {
            class_names[i]: (
                round(100.0 * per_class_correct[i] / per_class_total[i], 2)
                if per_class_total[i] > 0
                else 0
            )
            for i in range(len(class_names))
        },
        "config": {
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "max_neighbors_per_hop": args.max_neighbors_per_hop,
            "max_hops": args.max_hops,
            "denoising_steps": args.denoising_steps,
            "use_topology_mask": args.use_topology_mask,
            "lora_path": args.lora_path,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(args.log_file, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    logger.info("Results logged to %s", args.log_file)


if __name__ == "__main__":
    main()
