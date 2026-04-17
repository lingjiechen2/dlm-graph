"""
Infill-based evaluation: iterative denoising via MDLMSampler.infill().

Masks answer token positions, runs multi-step confidence-ranked unmasking,
then decodes predicted tokens and matches against numeric class labels.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_infill.py \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --dataset_name cora --max_hops 0 --steps 10

    CUDA_VISIBLE_DEVICES=0 python examples/tmdlm/eval_infill.py \
        --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
        --dataset_name ogbn-arxiv --max_hops 0 --steps 10 --max_answer_tokens 2
"""

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import torch
import transformers
from tqdm import tqdm

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.data.graph import (
    load_tag_dataset,
    get_answer_labels,
    DATASET_CONFIGS,
    get_class_token_ids,
)

logger = dllm.utils.get_default_logger(__name__)


_DIGIT_PREFIX_RE = re.compile(r"^(?:option\s*)?\d+\s*[):.\-]\s*")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize_pred(s: str) -> str:
    """Lowercase, strip leading 'N) ' / 'option N:' digit-prefixes and quotes."""
    s = s.lower().strip()
    s = _DIGIT_PREFIX_RE.sub("", s)
    return s.strip(" '\"\t")


def match_class(decoded: str, answer_labels: list) -> int:
    """
    Match decoded string to class index.

    Tries (in order):
      1. exact match after prefix/case normalization
      2. a label is contained in decoded (longest label wins on ties)
      3. decoded is contained in a label (only if unambiguous)
      4. distinguishing-token overlap (unique winner required)
    Returns -1 if no confident match.
    """
    dec = _normalize_pred(decoded)
    if not dec:
        return -1
    # 1. exact
    for k, lbl in enumerate(answer_labels):
        if dec == lbl:
            return k
    # 2. label is substring of decoded (most specific wins)
    hits = [k for k, lbl in enumerate(answer_labels) if lbl and lbl in dec]
    if hits:
        hits.sort(key=lambda k: -len(answer_labels[k]))
        return hits[0]
    # 3. decoded is substring of label (only if unique)
    hits = [k for k, lbl in enumerate(answer_labels) if dec in lbl]
    if len(hits) == 1:
        return hits[0]
    # 4. distinguishing-token overlap
    dec_toks = set(_TOKEN_RE.findall(dec))
    if not dec_toks:
        return -1
    all_label_toks = [set(_TOKEN_RE.findall(lbl)) for lbl in answer_labels]
    scores = []
    for k, lbl_toks in enumerate(all_label_toks):
        shared_across = set().union(
            *(t for j, t in enumerate(all_label_toks) if j != k)
        )
        distinguishing = lbl_toks - shared_across
        d_match = len(dec_toks & distinguishing)
        scores.append((d_match, len(dec_toks & lbl_toks), k))
    scores.sort(reverse=True)
    # require at least one distinguishing-token match, with a strict lead
    if scores[0][0] >= 1 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
        return scores[0][2]
    return -1


@dataclass
class EvalInfillArgs:
    exp: str = field(default="infill_eval", metadata={"help": "Experiment name"})
    model_name_or_path: str = field(default="GSAI-ML/LLaDA-8B-Instruct")
    dataset_name: str = field(default="cora")
    split: str = field(default="test")
    batch_size: int = field(default=8)
    max_seq_len: int = field(default=2048)
    max_neighbors_per_hop: int = field(default=10)
    max_hops: int = field(default=2)
    steps: int = field(
        default=10, metadata={"help": "Number of iterative denoising steps"}
    )
    temperature: float = field(
        default=0.0, metadata={"help": "Gumbel noise temperature (0 = greedy argmax)"}
    )
    remasking: str = field(
        default="low_confidence",
        metadata={
            "help": "Remasking strategy: low_confidence | random",
            "choices": ["low_confidence", "random"],
        },
    )
    log_file: str = field(default="experiments/experiment_log.jsonl")
    lora_path: str | None = field(
        default=None, metadata={"help": "Path to LoRA adapter checkpoint"}
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
    prompt_format: str = field(
        default="mc_digit",
        metadata={
            "help": "Prompt format: mc_digit | category_infill",
            "choices": ["mc_digit", "category_infill"],
        },
    )
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "Wrap prompt in LLaDA-Instruct chat template"},
    )
    include_neighbor_labels: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, prefix each neighbor with its ground-truth class name, "
                "e.g. 'Neighbor 1 [Neural Networks]: ...'. Oracle feature for "
                "test-time ablation — train does not see this."
            )
        },
    )
    include_options: bool = field(
        default=True,
        metadata={
            "help": (
                "If True (default), include 'Options: 0) ... 1) ...' list in the "
                "category_infill prompt. Set False for pure open-ended generation "
                "(the 'openended' setting in the README sweep)."
            )
        },
    )


@torch.no_grad()
def evaluate_with_sampler(
    model,
    tokenizer,
    test_dataset,
    num_classes,
    class_names,
    batch_size=8,
    steps=10,
    temperature=0.0,
    remasking="low_confidence",
    prompt_format="mc_digit",
):
    """
    Evaluate using MDLMSampler.infill() for iterative denoising.

    For each batch:
      1. Mask answer positions with mask_token_id
      2. Call infill() to iteratively denoise (steps passed to sampler)
      3. Read predicted tokens at answer positions
      4. Decode and match against:
         - numeric answer labels ("0", "1", ..., "N-1") for mc_digit
         - class name strings for category_infill (case-insensitive substring match)

    Returns:
        accuracy, per_class_correct, per_class_total, predictions, decoded_answers
    """
    device = next(model.parameters()).device

    if prompt_format == "mc_digit":
        answer_labels = get_answer_labels(num_classes)
    else:
        # category_infill: match against class names
        answer_labels = [name.lower().strip() for name in class_names]

    # Build sampler
    scheduler = LinearAlphaScheduler()
    sampler = MDLMSampler(model=model, tokenizer=tokenizer, scheduler=scheduler)
    sampler_config = MDLMSamplerConfig(
        steps=steps,
        temperature=temperature,
        remasking=remasking,
        block_size=8192,  # single block covering entire sequence
    )

    correct = 0
    total = 0
    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}
    all_predictions = []
    all_decoded = []
    no_match_count = 0

    # Process in batches manually (infill expects list of 1D tensors)
    num_samples = len(test_dataset)
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_samples = [test_dataset[i] for i in range(start_idx, end_idx)]
        b = len(batch_samples)

        # Collect metadata
        cls_labels = [s["cls_label"] for s in batch_samples]
        label_positions = [s["label_token_pos"] for s in batch_samples]
        answer_lens = [s.get("answer_len", 1) for s in batch_samples]

        # Build masked inputs (list of 1D tensors for infill)
        masked_inputs = []
        for idx, s in enumerate(batch_samples):
            ids = list(s["input_ids"])
            pos = label_positions[idx]
            ans_len = answer_lens[idx]
            for j in range(ans_len):
                if pos + j < len(ids):
                    ids[pos + j] = tokenizer.mask_token_id
            masked_inputs.append(ids)

        # Run infill
        denoised = sampler.infill(
            inputs=masked_inputs,
            config=sampler_config,
        )  # [b, T] tensor

        # Extract predictions at answer positions
        for i in range(b):
            pos = label_positions[i]
            ans_len = answer_lens[i]
            pred_tokens = denoised[i, pos : pos + ans_len].cpu().tolist()

            # Decode and match against answer labels
            decoded = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            all_decoded.append(decoded)

            pred_class = -1
            if prompt_format == "mc_digit":
                for k, label in enumerate(answer_labels):
                    if decoded == label:
                        pred_class = k
                        break
            else:
                pred_class = match_class(decoded, answer_labels)
            all_predictions.append(pred_class)

            gt = cls_labels[i]
            per_class_total[gt] += 1
            if pred_class == gt:
                correct += 1
                per_class_correct[gt] += 1
            if pred_class == -1:
                no_match_count += 1
            total += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0

    if no_match_count > 0:
        logger.warning(
            "%d/%d predictions did not match any class token", no_match_count, total
        )

    return accuracy, per_class_correct, per_class_total, all_predictions, all_decoded


def main():
    parser = transformers.HfArgumentParser(EvalInfillArgs)
    (args,) = parser.parse_args_into_dataclasses()

    # Resolve model path
    model_path = dllm.utils.resolve_with_base_env(
        args.model_name_or_path, "BASE_MODELS_DIR"
    )
    logger.info("Loading model from %s ...", model_path)

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

    # Get class info
    num_classes = DATASET_CONFIGS[args.dataset_name]["num_classes"]
    class_names, class_token_ids = get_class_token_ids(
        args.dataset_name,
        tokenizer,
        max_answer_tokens=args.max_answer_tokens,
        prompt_format=args.prompt_format,
    )
    # For category_infill, auto-compute max_answer_tokens from class token lists
    if args.prompt_format == "category_infill":
        args.max_answer_tokens = len(class_token_ids[0])
        logger.info(
            "category_infill: max_answer_tokens auto-set to %d",
            args.max_answer_tokens,
        )
    logger.info("Class names: %s", class_names)
    logger.info("Prompt format: %s", args.prompt_format)

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
        include_neighbor_labels=args.include_neighbor_labels,
        include_options=args.include_options,
    )
    logger.info("Loaded %d samples", len(test_dataset))

    # Print sample for sanity check
    sample = test_dataset[0]
    logger.info(
        "Sample 0: seq_len=%d, cls_label=%d, label_pos=%d, answer_len=%d",
        len(sample["input_ids"]),
        sample["cls_label"],
        sample["label_token_pos"],
        sample.get("answer_len", 1),
    )

    # Run evaluation
    t_start = time.time()
    accuracy, per_class_correct, per_class_total, predictions, decoded_answers = (
        evaluate_with_sampler(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            num_classes=num_classes,
            class_names=class_names,
            batch_size=args.batch_size,
            steps=args.steps,
            temperature=args.temperature,
            remasking=args.remasking,
            prompt_format=args.prompt_format,
        )
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
    logger.info(
        "Config: steps=%d, temperature=%.2f, remasking=%s",
        args.steps,
        args.temperature,
        args.remasking,
    )

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

    # Show prediction distribution
    pred_counts = Counter(predictions)
    logger.info("Prediction distribution:")
    for cls_id in range(len(class_names)):
        logger.info("  %s: %d", class_names[cls_id], pred_counts.get(cls_id, 0))
    if -1 in pred_counts:
        logger.info("  <no match>: %d", pred_counts[-1])

    # Show first 10 decoded answers
    logger.info("First 10 decoded answers:")
    for i in range(min(10, len(decoded_answers))):
        gt = test_dataset[i]["cls_label"]
        logger.info(
            "  [%d] GT=%s, Pred='%s', Match=%s",
            i,
            class_names[gt],
            decoded_answers[i],
            "Y" if predictions[i] == gt else "N",
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
            "steps": args.steps,
            "temperature": args.temperature,
            "remasking": args.remasking,
            "lora_path": args.lora_path,
            "max_answer_tokens": args.max_answer_tokens,
            "prompt_layout": args.prompt_layout,
            "prompt_format": args.prompt_format,
            "use_chat_template": args.use_chat_template,
            "include_neighbor_labels": args.include_neighbor_labels,
            "include_options": args.include_options,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(args.log_file, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    logger.info("Results logged to %s", args.log_file)


if __name__ == "__main__":
    main()
