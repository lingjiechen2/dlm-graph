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
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import torch
import transformers
from tqdm import tqdm

import torch.nn.functional as F

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.data.graph import (
    load_tag_dataset,
    DATASET_CONFIGS,
    get_class_token_ids,
    get_answer_labels,
)
from dllm.pipelines.tmdlm.utils import GraphDataCollator

logger = dllm.utils.get_default_logger(__name__)


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
    answer_label_style: str = field(
        default="digit0",
        metadata={
            "help": "MC answer label style: digit0 | number1 | letter",
            "choices": ["digit0", "number1", "letter"],
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
    neighbor_label_format: str = field(
        default="bracket",
        metadata={
            "help": (
                "When include_neighbor_labels=True, controls the phrasing: "
                "bracket='Neighbor 1 [Class]: ', paren='Neighbor 1 (category: Class): ', "
                "sentence='Neighbor 1 is a Class paper: ', colon='Neighbor 1 — Class: '."
            ),
            "choices": ["bracket", "paren", "sentence", "colon"],
        },
    )
    max_new_tokens: int = field(
        default=0,
        metadata={
            "help": "Generation window length (# mask tokens placed in answer region). "
            "0 = auto (mc_digit: max_answer_tokens; category_infill: max(max_answer_tokens, 6)). "
            "Larger values let the model write past the class name (e.g. trailing punctuation / extra words)."
        },
    )
    use_topology_mask: bool = field(
        default=False,
        metadata={"help": "Apply topology mask to attention (custom denoising loop)."},
    )
    max_samples: int = field(
        default=0,
        metadata={"help": "Cap on eval samples (0 = use full test split)."},
    )


_PUNCT = set(list(",.;:!?\"'`()[]{}<>/\\|"))


def _normalize(s: str) -> str:
    """Lowercase, strip, remove punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = " ".join(s.split())
    return s


def match_prediction(
    decoded: str,
    class_names,
    num_classes,
    prompt_format,
    answer_labels,
):
    """
    Two-tier matching over an open-ended decoded generation.

    Returns:
        pred_class_strict: int in [-1, num_classes-1]  — digit-only hit counted as wrong
        pred_class_lenient: int in [-1, num_classes-1] — digit-only hit counted as correct

    Matching priority:
      1. Normalized class-name hit (exact or substring either direction)
      2. Answer-label hit ("0"/"1"/..., "1"/"2"/..., or "a"/"b"/...) depending on style
    """
    decoded_norm = _normalize(decoded)
    name_norms = [_normalize(n) for n in class_names]
    label_norms = [_normalize(lbl) for lbl in answer_labels]

    # Class-name hit (exact or substring)
    name_hit = -1
    for k, nm in enumerate(name_norms):
        if decoded_norm == nm:
            name_hit = k
            break
    if name_hit == -1:
        for k, nm in enumerate(name_norms):
            if nm and (nm in decoded_norm or decoded_norm in nm):
                name_hit = k
                break

    label_hit = -1
    for tok in decoded_norm.split():
        for k, lbl in enumerate(label_norms):
            if tok == lbl:
                label_hit = k
                break
        if label_hit != -1:
            break

    if prompt_format == "category_infill":
        pred_strict = name_hit
        pred_lenient = name_hit if name_hit != -1 else label_hit
    else:
        pred_strict = label_hit
        pred_lenient = label_hit if label_hit != -1 else name_hit

    return pred_strict, pred_lenient


@torch.no_grad()
def infill_with_attention_mask(
    model,
    scheduler,
    tokenizer,
    input_ids_1d,
    attention_mask_4d,
    steps,
    temperature=0.0,
    remasking="low_confidence",
):
    """
    Single-sample iterative denoising that mirrors MDLMSampler.infill()'s core loop
    but uses an externally-provided 4D attention mask (e.g. topology mask) at every
    forward step. No CFG, no block splitting, single block covering the sequence.

    Args:
        input_ids_1d: list[int] or 1D tensor — masked canvas
        attention_mask_4d: [1, 1, T, T] additive mask (0 or -inf)

    Returns:
        x: [1, T] tensor of denoised token ids
    """
    device = model.device
    mask_id = tokenizer.mask_token_id

    x = torch.as_tensor(input_ids_1d, dtype=torch.long, device=device).unsqueeze(0)
    B, T = x.shape
    mask_index = x == mask_id  # [1, T]

    num_transfer_tokens = get_num_transfer_tokens(
        mask_index=mask_index,
        steps=steps,
        scheduler=scheduler,
        stochastic=False,
    )
    effective_steps = num_transfer_tokens.size(1)

    for s in range(effective_steps):
        mask_index_full = x == mask_id
        if not mask_index_full.any():
            break

        logits = model(x, attention_mask=attention_mask_4d).logits  # [1, T, V]
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # [1, T]

        if remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [1, T]
        elif remasking == "random":
            x0_p = torch.rand((B, T), device=device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index_full, x0, x)
        confidence = torch.where(
            mask_index_full, x0_p, torch.full_like(x0_p, -float("inf"))
        )

        k = int(num_transfer_tokens[0, s].item())
        if k > 0:
            _, select_idx = torch.topk(confidence[0], k=k)
            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            transfer_index[0, select_idx] = True
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def evaluate_with_sampler(
    model,
    tokenizer,
    test_dataset,
    num_classes,
    class_names,
    max_answer_tokens,
    max_new_tokens,
    steps=10,
    temperature=0.0,
    remasking="low_confidence",
    prompt_format="mc_digit",
    answer_label_style="digit0",
    use_topology_mask=False,
    max_samples=0,
):
    """
    Open-ended evaluation via MDLMSampler.infill().

    Per sample:
      1. Replace the sample's ``max_answer_tokens`` reserved answer slots with
         ``max_new_tokens`` mask tokens (a fixed generation window).
      2. Run iterative denoising.
      3. Decode the entire generation window (not just answer_len).
      4. Normalize (lowercase, strip punctuation) and match against class names;
         fall back to digit matching for the lenient score.

    Returns:
        (strict_acc, lenient_acc, per_class_correct_strict, per_class_total,
         predictions_strict, predictions_lenient, decoded_answers)
    """

    scheduler = LinearAlphaScheduler()
    sampler = MDLMSampler(model=model, tokenizer=tokenizer, scheduler=scheduler)
    sampler_config = MDLMSamplerConfig(
        steps=steps,
        temperature=temperature,
        remasking=remasking,
        block_size=8192,
    )
    collator = (
        GraphDataCollator(tokenizer=tokenizer, padding=True, return_tensors="pt")
        if use_topology_mask
        else None
    )
    device = next(model.parameters()).device

    correct_strict = 0
    correct_lenient = 0
    total = 0
    per_class_correct_strict = {i: 0 for i in range(num_classes)}
    per_class_correct_lenient = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}
    all_strict = []
    all_lenient = []
    all_decoded = []
    answer_labels = get_answer_labels(num_classes, style=answer_label_style)
    no_match_count = 0

    num_samples = len(test_dataset)
    if max_samples and max_samples > 0:
        num_samples = min(num_samples, max_samples)
    pbar = tqdm(range(num_samples), desc="Evaluating")
    for idx in pbar:
        s = test_dataset[idx]
        ids = list(s["input_ids"])
        pos = s["label_token_pos"]
        gt = s["cls_label"]

        # Determine reserve length (# slots the sample has at pos for the answer)
        if prompt_format == "category_infill":
            reserve = max_answer_tokens
        else:
            # mc_digit: no right-padding; reserve equals real answer_len
            reserve = s.get("answer_len", 1)

        # Replace [pos : pos + reserve] with max_new_tokens mask tokens
        prefix = ids[:pos]
        suffix = ids[pos + reserve :]
        gen_window = [tokenizer.mask_token_id] * max_new_tokens
        masked_ids = prefix + gen_window + suffix
        gen_start = len(prefix)
        gen_end = gen_start + max_new_tokens

        # Shift spans in the sample by (max_new_tokens - reserve) for positions
        # past pos (needed for topology mask construction over the new length)
        shift = max_new_tokens - reserve
        patched_sample = None
        if use_topology_mask:
            patched_sample = {
                "input_ids": masked_ids,
                "labels": [-100] * len(masked_ids),
                "node_spans": [
                    [
                        st if st < pos else st + shift,
                        en if en <= pos else en + shift,
                    ]
                    for (st, en) in s.get("node_spans", [[0, len(ids)]])
                ],
                "node_hops": s.get("node_hops", [0]),
                "cls_label": gt,
                "label_token_pos": pos,
                "answer_len": max_new_tokens,
            }

        if use_topology_mask:
            batch = collator([patched_sample])
            topo = batch["topology_mask"].to(device)  # [1, T, T]
            T_len = topo.shape[-1]
            additive = torch.zeros(1, T_len, T_len, device=device, dtype=model.dtype)
            additive = additive.masked_fill(topo == 0, float("-inf"))
            attn_mask_4d = additive.unsqueeze(1)  # [1, 1, T, T]
            # The collator pads to its own max_len; input_ids may be padded too.
            canvas = batch["input_ids"].to(device)[0].tolist()
            denoised = infill_with_attention_mask(
                model=model,
                scheduler=scheduler,
                tokenizer=tokenizer,
                input_ids_1d=canvas,
                attention_mask_4d=attn_mask_4d,
                steps=steps,
                temperature=temperature,
                remasking=remasking,
            )  # [1, T]
        else:
            denoised = sampler.infill(
                inputs=[masked_ids],
                config=sampler_config,
            )  # [1, T]

        pred_tokens = denoised[0, gen_start:gen_end].cpu().tolist()
        decoded = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        all_decoded.append(decoded)

        pred_strict, pred_lenient = match_prediction(
            decoded,
            class_names,
            num_classes,
            prompt_format,
            answer_labels,
        )
        all_strict.append(pred_strict)
        all_lenient.append(pred_lenient)

        per_class_total[gt] += 1
        if pred_strict == gt:
            correct_strict += 1
            per_class_correct_strict[gt] += 1
        if pred_lenient == gt:
            correct_lenient += 1
            per_class_correct_lenient[gt] += 1
        if pred_strict == -1 and pred_lenient == -1:
            no_match_count += 1
        total += 1

        running_strict = 100.0 * correct_strict / total if total > 0 else 0.0
        running_lenient = 100.0 * correct_lenient / total if total > 0 else 0.0
        pbar.set_postfix(
            strict=f"{running_strict:.2f}%",
            lenient=f"{running_lenient:.2f}%",
        )

    strict_acc = 100.0 * correct_strict / total if total > 0 else 0.0
    lenient_acc = 100.0 * correct_lenient / total if total > 0 else 0.0

    if no_match_count > 0:
        logger.warning(
            "%d/%d generations matched neither a class name nor a digit",
            no_match_count,
            total,
        )

    return (
        strict_acc,
        lenient_acc,
        per_class_correct_strict,
        per_class_correct_lenient,
        per_class_total,
        all_strict,
        all_lenient,
        all_decoded,
    )


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
        answer_label_style=args.answer_label_style,
    )
    # For category_infill, optionally auto-compute max_answer_tokens.
    # Keep explicit user value when > 0 so eval can match train-time reserve.
    if args.prompt_format == "category_infill" and args.max_answer_tokens <= 0:
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
        answer_label_style=args.answer_label_style,
        include_neighbor_labels=args.include_neighbor_labels,
        neighbor_label_format=args.neighbor_label_format,
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

    # Resolve generation window length
    if args.max_new_tokens <= 0:
        if args.prompt_format == "category_infill":
            args.max_new_tokens = max(args.max_answer_tokens, 6)
        else:
            args.max_new_tokens = args.max_answer_tokens
    logger.info(
        "Generation window: max_new_tokens=%d (answer reserve=%d)",
        args.max_new_tokens,
        args.max_answer_tokens,
    )

    # Run evaluation
    t_start = time.time()
    (
        strict_acc,
        lenient_acc,
        per_class_correct_strict,
        per_class_correct_lenient,
        per_class_total,
        preds_strict,
        preds_lenient,
        decoded_answers,
    ) = evaluate_with_sampler(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_classes=num_classes,
        class_names=class_names,
        max_answer_tokens=args.max_answer_tokens,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        remasking=args.remasking,
        prompt_format=args.prompt_format,
        answer_label_style=args.answer_label_style,
        use_topology_mask=args.use_topology_mask,
        max_samples=args.max_samples,
    )
    elapsed = time.time() - t_start

    total_cnt = sum(per_class_total.values())

    # Log results
    logger.info("=" * 60)
    logger.info("Experiment: %s", args.exp)
    logger.info(
        "Strict Accuracy:  %.2f%% (%d/%d)  [class-name match only]",
        strict_acc,
        sum(per_class_correct_strict.values()),
        total_cnt,
    )
    logger.info(
        "Lenient Accuracy: %.2f%% (%d/%d)  [class-name or digit match]",
        lenient_acc,
        sum(per_class_correct_lenient.values()),
        total_cnt,
    )
    logger.info("Time: %.1f seconds", elapsed)
    logger.info(
        "Config: steps=%d, temperature=%.2f, remasking=%s, max_new_tokens=%d",
        args.steps,
        args.temperature,
        args.remasking,
        args.max_new_tokens,
    )

    for i, name in enumerate(class_names):
        t = per_class_total[i]
        cs = per_class_correct_strict[i]
        cl = per_class_correct_lenient[i]
        logger.info(
            "  Class %d (%s): strict=%.1f%% lenient=%.1f%% (%d/%d, %d/%d)",
            i,
            name,
            100.0 * cs / t if t > 0 else 0,
            100.0 * cl / t if t > 0 else 0,
            cs,
            t,
            cl,
            t,
        )

    # Strict prediction distribution
    pred_counts = Counter(preds_strict)
    logger.info("Prediction distribution (strict):")
    for cls_id in range(len(class_names)):
        logger.info("  %s: %d", class_names[cls_id], pred_counts.get(cls_id, 0))
    if -1 in pred_counts:
        logger.info("  <no match>: %d", pred_counts[-1])

    # Show first 10 decoded answers
    logger.info("First 10 decoded answers:")
    for i in range(min(10, len(decoded_answers))):
        gt = test_dataset[i]["cls_label"]
        logger.info(
            "  [%d] GT=%s, Pred='%s', Strict=%s, Lenient=%s",
            i,
            class_names[gt],
            decoded_answers[i],
            "Y" if preds_strict[i] == gt else "N",
            "Y" if preds_lenient[i] == gt else "N",
        )

    # Save to log file
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": args.exp,
        "model": args.model_name_or_path,
        "dataset": args.dataset_name,
        "split": args.split,
        "accuracy_strict": round(strict_acc, 2),
        "accuracy_lenient": round(lenient_acc, 2),
        "per_class_accuracy_strict": {
            class_names[i]: (
                round(100.0 * per_class_correct_strict[i] / per_class_total[i], 2)
                if per_class_total[i] > 0
                else 0
            )
            for i in range(len(class_names))
        },
        "per_class_accuracy_lenient": {
            class_names[i]: (
                round(100.0 * per_class_correct_lenient[i] / per_class_total[i], 2)
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
            "max_new_tokens": args.max_new_tokens,
            "prompt_layout": args.prompt_layout,
            "prompt_format": args.prompt_format,
            "answer_label_style": args.answer_label_style,
            "use_chat_template": args.use_chat_template,
            "include_neighbor_labels": args.include_neighbor_labels,
            "neighbor_label_format": args.neighbor_label_format,
            "use_topology_mask": args.use_topology_mask,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(args.log_file, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    logger.info("Results logged to %s", args.log_file)


if __name__ == "__main__":
    main()
