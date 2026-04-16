"""
Logit-based evaluation: single forward pass + restricted argmax / log-prob scoring.

Masks answer positions, runs a SINGLE forward pass, then scores each
candidate class by its logit at the answer position(s). No iterative denoising
— for multi-step natural generation, use eval_infill.py instead, which passes
the step count directly into MDLMSampler.

Usage
-----
Single-token (restricted argmax):
    CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_logit.py \
        --exp baseline_cora \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --dataset_name cora --max_hops 0

Multi-token (mean log-prob):
    CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_logit.py \
        --exp baseline_arxiv \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --dataset_name ogbn-arxiv --max_hops 0 --max_answer_tokens 2
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
from dllm.data.graph import load_tag_dataset, get_class_token_ids
from dllm.pipelines.tmdlm.utils import GraphDataCollator

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class EvalLogitArgs:
    exp: str = field(metadata={"help": "Experiment name for logging"})
    model_name_or_path: str = field(default="GSAI-ML/LLaDA-8B-Instruct")
    dataset_name: str = field(default="cora")
    split: str = field(default="test")
    batch_size: int = field(default=8)
    max_seq_len: int = field(default=2048)
    max_neighbors_per_hop: int = field(default=10)
    max_hops: int = field(default=2)
    use_topology_mask: bool = field(
        default=False, metadata={"help": "Apply topology mask to attention"}
    )
    log_file: str = field(default="experiments/experiment_log.jsonl")
    lora_path: str | None = field(
        default=None, metadata={"help": "Path to LoRA adapter checkpoint"}
    )
    position_id_type: str = field(
        default="sequential",
        metadata={
            "help": "Position ID scheme: sequential | topological",
            "choices": ["sequential", "topological"],
        },
    )
    seed: int = field(default=42)
    max_answer_tokens: int = field(
        default=1,
        metadata={"help": "Max answer tokens per class (1=single digit, 2=two-digit)"},
    )
    prompt_layout: str = field(
        default="target_first",
        metadata={
            "help": "Prompt layout: target_first | neighbor_first",
            "choices": ["target_first", "neighbor_first"],
        },
    )
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "Wrap prompt in LLaDA-Instruct chat template"},
    )
    prompt_format: str = field(
        default="mc_digit",
        metadata={
            "help": "Prompt format: mc_digit | category_infill",
            "choices": ["mc_digit", "category_infill"],
        },
    )


@torch.no_grad()
def evaluate_layer0(
    model,
    tokenizer,
    test_dataset,
    class_first_token_ids,
    batch_size=8,
    use_topology_mask=False,
    position_id_type="sequential",
    max_answer_tokens=1,
):
    """
    Layer 0 evaluation: frozen model, mask answer tokens, single forward pass,
    classify by restricted argmax (single-token) or log-prob sum (multi-token).

    Returns:
        accuracy: float
        per_class_correct: dict[int, int]
        per_class_total: dict[int, int]
        predictions: list[int]
    """
    device = next(model.parameters()).device

    num_classes = len(class_first_token_ids)
    if max_answer_tokens == 1:
        class_token_ids_tensor = torch.tensor(class_first_token_ids, device=device)
    else:
        # class_first_token_ids is list[list[int]], shape [K, max_answer_tokens]
        class_token_ids_tensor = torch.tensor(
            class_first_token_ids, device=device, dtype=torch.long
        )  # [K, max_answer_tokens]

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
        label_indices = batch["label_token_indices"]  # [b, max_ans_len]
        b, l = input_ids.shape

        # Mask answer tokens: use labels != -100 for single-token mode,
        # but for multi-token mode, mask ALL answer positions (including padding)
        # to prevent answer-length leakage.
        masked_input = input_ids.clone()
        if max_answer_tokens == 1:
            maskable = labels != -100  # [b, l]
            masked_input[maskable] = tokenizer.mask_token_id
        else:
            # Mask all answer positions to prevent the model from seeing
            # which positions are pad vs real tokens (leaks answer length)
            maskable = labels != -100
            masked_input[maskable] = tokenizer.mask_token_id
            for j in range(label_indices.shape[1]):
                masked_input[torch.arange(b, device=device), label_indices[:, j]] = (
                    tokenizer.mask_token_id
                )

        # Build attention mask
        if use_topology_mask and "topology_mask" in batch:
            topo = batch["topology_mask"]  # [b, l, l]
            additive = torch.zeros(b, l, l, device=device, dtype=model.dtype)
            additive = additive.masked_fill(topo == 0, float("-inf"))
            attn_mask = additive.unsqueeze(1)  # [b, 1, l, l]
        else:
            # Standard padding-only mask (full attention among valid tokens)
            attn_mask = batch.get("attention_mask", None)  # [b, l]

        # Single forward pass
        pos_ids = batch.get("position_ids", None)
        forward_kwargs = dict(input_ids=masked_input, attention_mask=attn_mask)
        if pos_ids is not None:
            forward_kwargs["position_ids"] = pos_ids
        outputs = model(**forward_kwargs)
        logits = outputs.logits  # [b, l, V]

        if max_answer_tokens == 1:
            # Single-token: restricted argmax over class token IDs
            label_logits = logits[
                torch.arange(b, device=device).unsqueeze(1),
                label_indices,
            ].squeeze(
                1
            )  # [b, V]
            restricted_logits = label_logits[:, class_token_ids_tensor]  # [b, K]
            preds = restricted_logits.argmax(dim=-1)  # [b]
        else:
            # Multi-token: sum log-probs across answer positions per class
            # label_indices: [b, max_ans_len]
            # class_token_ids_tensor: [K, max_ans_len]
            ans_len = label_indices.shape[1]
            log_probs = F.log_softmax(logits, dim=-1)  # [b, l, V]

            # Gather log-probs at answer positions: [b, max_ans_len, V]
            pos_expanded = label_indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
            ans_log_probs = log_probs.gather(1, pos_expanded)  # [b, max_ans_len, V]

            # For each class k, gather log-probs for its tokens: [b, K]
            # class_token_ids_tensor: [K, max_ans_len] -> expand to [b, K, max_ans_len]
            K = class_token_ids_tensor.shape[0]
            cls_ids = class_token_ids_tensor.unsqueeze(0).expand(
                b, -1, -1
            )  # [b, K, max_ans_len]

            # ans_log_probs: [b, max_ans_len, V] -> for each position, gather the K class tokens
            scores = torch.zeros(b, K, device=device)
            # Build validity mask: position j is valid if token != pad_token_id
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            cls_valid = class_token_ids_tensor != pad_id  # [K, max_ans_len]

            for j in range(ans_len):
                pos_logprobs = ans_log_probs[:, j, :]  # [b, V]
                cls_tokens_j = cls_ids[:, :, j]  # [b, K]
                token_scores = pos_logprobs.gather(1, cls_tokens_j)  # [b, K]
                # Only add score if this position is valid for the class
                valid_j = cls_valid[:, j].unsqueeze(0).expand(b, -1)  # [b, K]
                scores += token_scores * valid_j.float()

            # Normalize by number of valid tokens per class (mean log-prob)
            cls_lengths = cls_valid.sum(dim=1).float().clamp(min=1)  # [K]
            scores = scores / cls_lengths.unsqueeze(0)  # [b, K]

            preds = scores.argmax(dim=-1)  # [b]

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
    parser = transformers.HfArgumentParser(EvalLogitArgs)
    (args,) = parser.parse_args_into_dataclasses()

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
        args.dataset_name,
        tokenizer,
        max_answer_tokens=args.max_answer_tokens,
        prompt_format=args.prompt_format,
    )

    # For category_infill, auto-compute max_answer_tokens from returned token IDs
    if args.prompt_format == "category_infill":
        # class_first_token_ids is list[list[int]], infer max_answer_tokens
        args.max_answer_tokens = len(class_first_token_ids[0])
        logger.info("category_infill: max_answer_tokens=%d", args.max_answer_tokens)

    logger.info("Class names: %s", class_names)
    logger.info("Class token IDs: %s", class_first_token_ids)

    # Verify token IDs are unique (only for single-token mode)
    if args.max_answer_tokens == 1:
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
        max_answer_tokens=args.max_answer_tokens,
        prompt_layout=args.prompt_layout,
        use_chat_template=args.use_chat_template,
        prompt_format=args.prompt_format,
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
        position_id_type=args.position_id_type,
        max_answer_tokens=args.max_answer_tokens,
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
            "use_topology_mask": args.use_topology_mask,
            "lora_path": args.lora_path,
            "prompt_layout": args.prompt_layout,
            "use_chat_template": args.use_chat_template,
            "prompt_format": args.prompt_format,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(args.log_file, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    logger.info("Results logged to %s", args.log_file)


if __name__ == "__main__":
    main()
