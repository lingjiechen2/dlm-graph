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


def _valid_class_token_lists(class_token_ids, tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return [[tok for tok in seq if tok != pad_id] for seq in class_token_ids]


def _shared_prefix_len(token_lists):
    if not token_lists:
        return 0
    prefix_len = 0
    min_len = min(len(seq) for seq in token_lists)
    while prefix_len < min_len:
        tok = token_lists[0][prefix_len]
        if all(seq[prefix_len] == tok for seq in token_lists[1:]):
            prefix_len += 1
        else:
            break
    return prefix_len


def _predict_pubmed_hierarchical(
    log_probs,
    label_starts,
    class_names,
    class_token_ids_tensor,
    tokenizer,
):
    """Hierarchical PubMed scorer.

    Treat the shared prefix "Diabetes Mellitus" as prompt context. First compare
    the next token for Experimental (",") against the Type branch ("Type"). If
    Type wins, compare the following token between Type 1 and Type 2.
    """
    device = log_probs.device
    token_lists = _valid_class_token_lists(class_token_ids_tensor.tolist(), tokenizer)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    exp_idx = name_to_idx["Diabetes Mellitus, Experimental"]
    type1_idx = name_to_idx["Diabetes Mellitus Type 1"]
    type2_idx = name_to_idx["Diabetes Mellitus Type 2"]

    shared_len = _shared_prefix_len(token_lists)
    type_shared_len = _shared_prefix_len([token_lists[type1_idx], token_lists[type2_idx]])

    exp_branch_tok = token_lists[exp_idx][shared_len]
    type_branch_tok = token_lists[type1_idx][shared_len]
    type1_leaf_tok = token_lists[type1_idx][type_shared_len]
    type2_leaf_tok = token_lists[type2_idx][type_shared_len]

    branch_pos = shared_len
    leaf_pos = type_shared_len

    branch_abs_pos = label_starts + branch_pos
    branch_scores = log_probs[
        torch.arange(log_probs.shape[0], device=device),
        branch_abs_pos,
        :,
    ][:, [exp_branch_tok, type_branch_tok]]
    choose_exp = branch_scores[:, 0] >= branch_scores[:, 1]

    preds = torch.full((log_probs.shape[0],), type1_idx, device=device, dtype=torch.long)
    preds[choose_exp] = exp_idx

    if (~choose_exp).any():
        type_rows = (~choose_exp).nonzero(as_tuple=False).squeeze(-1)
        leaf_abs_pos = label_starts[type_rows] + leaf_pos
        leaf_scores = log_probs[
            type_rows,
            leaf_abs_pos,
            :,
        ][:, [type1_leaf_tok, type2_leaf_tok]]
        preds[type_rows] = torch.where(
            leaf_scores[:, 0] >= leaf_scores[:, 1],
            torch.full_like(type_rows, type1_idx),
            torch.full_like(type_rows, type2_idx),
        )

    return preds


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
    answer_label_style: str = field(
        default="digit0",
        metadata={
            "help": "MC answer label style: digit0 | digit0_pad | number1 | letter",
            "choices": ["digit0", "digit0_pad", "number1", "letter"],
        },
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
    max_samples: int = field(
        default=0,
        metadata={"help": "Limit eval to first N samples (0 = all)"},
    )
    apply_class_prior_calibration: bool = field(
        default=False,
        metadata={"help": "If True, also report calibrated accuracy = argmax(logit - log p_train(c))"},
    )
    train_resample_strategy: str = field(
        default="none",
        metadata={"help": "Used only with apply_class_prior_calibration. Should match SFT-time value (none/boost/balance_classes)"},
    )
    train_boost_spec: str = field(
        default="",
        metadata={"help": "Used only with apply_class_prior_calibration. Should match SFT-time value"},
    )
    train_max_train_samples: int = field(
        default=0,
        metadata={"help": "Used only with apply_class_prior_calibration. Should match SFT-time max_train_samples (0 = full train)"},
    )
    dump_logits_path: str | None = field(
        default=None,
        metadata={"help": "If set, save per-sample scores / per-position log-probs / cls_labels / log_prior to this .npz path for offline post-processing."},
    )
    dump_per_position: bool = field(
        default=False,
        metadata={"help": "When dump_logits_path is set, also save per-answer-position class log-probs (shape [N, max_ans_len, K]). Multi-token only."},
    )
    neighbor_seed: int = field(
        default=-1,
        metadata={"help": "If >=0, governs neighbor sampling RNG independently of `seed` (which controls the 1000-sample selection). Use to do TTA over fixed sample IDs with jittered neighbors."},
    )


@torch.no_grad()
def evaluate_layer0(
    model,
    tokenizer,
    class_names,
    test_dataset,
    class_first_token_ids,
    batch_size=8,
    use_topology_mask=False,
    position_id_type="sequential",
    max_answer_tokens=1,
    dataset_name="cora",
    prompt_format="mc_digit",
    log_prior=None,
    dump_scores=False,
    dump_per_position=False,
):
    """
    Layer 0 evaluation: frozen model, mask answer tokens, single forward pass,
    classify by restricted argmax (single-token) or log-prob sum (multi-token).

    If log_prior is provided ([K] tensor), also computes calibrated predictions
    via argmax(scores - log_prior) and tracks them in parallel.

    Returns dict with raw and calibrated metrics.
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
    correct_cal = 0
    total = 0
    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_correct_cal = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}
    all_predictions = []
    all_predictions_cal = []
    all_scores = [] if dump_scores else None  # list of [b, K] tensors
    all_per_pos = [] if (dump_scores and dump_per_position) else None  # list of [b, max_ans_len, K]
    all_cls_labels = [] if dump_scores else None

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        input_ids = batch["input_ids"]  # [b, l]
        labels = batch["labels"]  # [b, l]
        cls_labels = batch["cls_labels"]  # [b]
        label_indices = batch["label_token_indices"]  # [b, max_ans_len]
        b, l = input_ids.shape

        # Build the masked eval prompt. The answer window at
        # label_indices[:, :] is overwritten with mask tokens, and attention
        # is forced to 1 on those positions (the collator zeroed it on
        # reserved-tail pads for training). Result is byte-identical to the
        # prompt eval_infill constructs; the only legitimate difference lives
        # in how we read the model output below (single-pass restricted argmax
        # vs iterative diffusion + text match there).
        masked_input = input_ids.clone()
        b_arange = torch.arange(b, device=device)
        attn1d = batch.get("attention_mask", None)
        if attn1d is not None:
            attn1d = attn1d.clone()
        for j in range(label_indices.shape[1]):
            masked_input[b_arange, label_indices[:, j]] = tokenizer.mask_token_id
            if attn1d is not None:
                attn1d[b_arange, label_indices[:, j]] = 1
        if use_topology_mask and "topology_mask" in batch:
            topo = batch["topology_mask"]  # [b, l, l]
            additive = torch.zeros(b, l, l, device=device, dtype=model.dtype)
            additive = additive.masked_fill(topo == 0, float("-inf"))
            attn_mask = additive.unsqueeze(1)  # [b, 1, l, l]
        else:
            attn_mask = attn1d

        # Single forward pass
        pos_ids = batch.get("position_ids", None)
        forward_kwargs = dict(input_ids=masked_input, attention_mask=attn_mask)
        if pos_ids is not None:
            forward_kwargs["position_ids"] = pos_ids
        outputs = model(**forward_kwargs)
        logits = outputs.logits  # [b, l, V]

        scores_for_dump = None
        per_pos_for_dump = None

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
            if log_prior is not None:
                preds_cal = (restricted_logits - log_prior).argmax(dim=-1)
            else:
                preds_cal = preds
            scores_for_dump = restricted_logits
        else:
            # Multi-token: sum log-probs across answer positions per class
            # label_indices: [b, max_ans_len]
            # class_token_ids_tensor: [K, max_ans_len]
            ans_len = label_indices.shape[1]
            log_probs = F.log_softmax(logits, dim=-1)  # [b, l, V]

            # Gather log-probs at answer positions: [b, max_ans_len, V]
            pos_expanded = label_indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
            ans_log_probs = log_probs.gather(1, pos_expanded)  # [b, max_ans_len, V]

            if dataset_name == "pubmed" and prompt_format == "category_infill":
                preds = _predict_pubmed_hierarchical(
                    log_probs=log_probs,
                    label_starts=label_indices[:, 0],
                    class_names=class_names,
                    class_token_ids_tensor=class_token_ids_tensor,
                    tokenizer=tokenizer,
                )
                preds_cal = preds  # calibration not implemented for hierarchical pubmed
            else:
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
                if log_prior is not None:
                    preds_cal = (scores - log_prior).argmax(dim=-1)
                else:
                    preds_cal = preds
                scores_for_dump = scores
                if all_per_pos is not None:
                    pp = torch.zeros(b, ans_len, K, device=device)
                    for j in range(ans_len):
                        pp[:, j, :] = ans_log_probs[:, j, :].gather(1, cls_ids[:, :, j])
                    per_pos_for_dump = pp

        if all_scores is not None and scores_for_dump is not None:
            all_scores.append(scores_for_dump.detach().cpu())
            all_cls_labels.append(cls_labels.detach().cpu())
        if all_per_pos is not None and per_pos_for_dump is not None:
            all_per_pos.append(per_pos_for_dump.detach().cpu())

        all_predictions.extend(preds.cpu().tolist())
        all_predictions_cal.extend(preds_cal.cpu().tolist())

        for i in range(b):
            gt = cls_labels[i].item()
            pred = preds[i].item()
            pred_cal = preds_cal[i].item()
            per_class_total[gt] += 1
            if pred == gt:
                correct += 1
                per_class_correct[gt] += 1
            if pred_cal == gt:
                correct_cal += 1
                per_class_correct_cal[gt] += 1
            total += 1

        running_acc = 100.0 * correct / total if total > 0 else 0.0
        running_acc_cal = 100.0 * correct_cal / total if total > 0 else 0.0
        pbar.set_postfix(acc=f"{running_acc:.2f}%", acc_cal=f"{running_acc_cal:.2f}%", n=total)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    accuracy_cal = 100.0 * correct_cal / total if total > 0 else 0.0
    out = {
        "accuracy": accuracy,
        "accuracy_calibrated": accuracy_cal,
        "per_class_correct": per_class_correct,
        "per_class_correct_calibrated": per_class_correct_cal,
        "per_class_total": per_class_total,
        "predictions": all_predictions,
        "predictions_calibrated": all_predictions_cal,
    }
    if all_scores is not None and len(all_scores) > 0:
        out["scores"] = torch.cat(all_scores, dim=0).numpy()  # [N, K]
        out["cls_labels"] = torch.cat(all_cls_labels, dim=0).numpy()  # [N]
        if all_per_pos is not None and len(all_per_pos) > 0:
            out["per_pos_logprobs"] = torch.cat(all_per_pos, dim=0).numpy()  # [N, max_ans_len, K]
    return out


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
        answer_label_style=args.answer_label_style,
    )

    # For category_infill, optionally auto-compute max_answer_tokens.
    # Keep explicit user value when > 0 so eval can match train-time reserve.
    if args.prompt_format == "category_infill" and args.max_answer_tokens <= 0:
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
        answer_label_style=args.answer_label_style,
        include_neighbor_labels=args.include_neighbor_labels,
        neighbor_label_format=args.neighbor_label_format,
        max_samples=args.max_samples,
        neighbor_seed=args.neighbor_seed if args.neighbor_seed >= 0 else None,
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

    # Optionally compute logit-adjustment shift for post-hoc calibration.
    # Calibrated score = logit - (log p_train - log p_test) = logit - log_prior_shift
    # where p_train is the SFT-time class distribution (boosted) and p_test is
    # the eval-time class distribution (estimated from the test set itself).
    # This corrects for both (a) SFT-time boost bias and (b) train/test class
    # distribution shift.
    log_prior = None
    if args.apply_class_prior_calibration:
        logger.info("Loading train split for class prior calibration (resample=%s, boost=%s, max_train_samples=%d)...",
                    args.train_resample_strategy, args.train_boost_spec, args.train_max_train_samples)
        train_dataset_for_prior = load_tag_dataset(
            args.dataset_name,
            tokenizer=tokenizer,
            split="train",
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
            max_samples=args.train_max_train_samples,
            resample_strategy=args.train_resample_strategy,
            boost_spec=args.train_boost_spec,
        )
        K = len(class_first_token_ids)
        train_counts = torch.zeros(K, dtype=torch.float64)
        for s in train_dataset_for_prior:
            train_counts[int(s["cls_label"])] += 1.0
        test_counts = torch.zeros(K, dtype=torch.float64)
        for s in test_dataset:
            test_counts[int(s["cls_label"])] += 1.0
        eps = 1.0  # Laplace smoothing
        train_probs = (train_counts + eps) / (train_counts.sum() + eps * K)
        test_probs = (test_counts + eps) / (test_counts.sum() + eps * K)
        log_train_prior = torch.log(train_probs)
        log_test_prior = torch.log(test_probs)
        # Subtracting this from logits applies the boost-correction + test-prior re-injection.
        log_prior = (log_train_prior - log_test_prior).to(device=device, dtype=torch.float32)
        logger.info("Train top-5 classes: %s",
                    [(int(i), class_names[int(i)], int(c)) for i, c in sorted(enumerate(train_counts.tolist()), key=lambda x: -x[1])[:5]])
        logger.info("Test  top-5 classes: %s",
                    [(int(i), class_names[int(i)], int(c)) for i, c in sorted(enumerate(test_counts.tolist()), key=lambda x: -x[1])[:5]])
        logger.info("Calibration shift (log p_train - log p_test) min=%.3f max=%.3f range=%.3f",
                    log_prior.min().item(), log_prior.max().item(), (log_prior.max()-log_prior.min()).item())

    # Run evaluation
    t_start = time.time()
    eval_out = evaluate_layer0(
        model=model,
        tokenizer=tokenizer,
        class_names=class_names,
        test_dataset=test_dataset,
        class_first_token_ids=class_first_token_ids,
        batch_size=args.batch_size,
        use_topology_mask=args.use_topology_mask,
        position_id_type=args.position_id_type,
        max_answer_tokens=args.max_answer_tokens,
        dataset_name=args.dataset_name,
        prompt_format=args.prompt_format,
        log_prior=log_prior,
        dump_scores=args.dump_logits_path is not None,
        dump_per_position=args.dump_per_position,
    )
    accuracy = eval_out["accuracy"]
    accuracy_cal = eval_out["accuracy_calibrated"]
    per_class_correct = eval_out["per_class_correct"]
    per_class_correct_cal = eval_out["per_class_correct_calibrated"]
    per_class_total = eval_out["per_class_total"]
    predictions = eval_out["predictions"]
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
    if log_prior is not None:
        logger.info(
            "Calibrated accuracy: %.2f%% (%d/%d)",
            accuracy_cal,
            sum(per_class_correct_cal.values()),
            sum(per_class_total.values()),
        )
    logger.info("Time: %.1f seconds", elapsed)

    for i, name in enumerate(class_names):
        c, t = per_class_correct[i], per_class_total[i]
        c_cal = per_class_correct_cal[i]
        if log_prior is not None:
            logger.info(
                "  Class %d (%s): raw %.1f%% (%d/%d) | cal %.1f%% (%d/%d)",
                i, name,
                100.0 * c / t if t > 0 else 0, c, t,
                100.0 * c_cal / t if t > 0 else 0, c_cal, t,
            )
        else:
            logger.info(
                "  Class %d (%s): %.1f%% (%d/%d)",
                i, name,
                100.0 * c / t if t > 0 else 0, c, t,
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
        "accuracy_calibrated": round(accuracy_cal, 2) if log_prior is not None else None,
        "per_class_accuracy": {
            class_names[i]: (
                round(100.0 * per_class_correct[i] / per_class_total[i], 2)
                if per_class_total[i] > 0
                else 0
            )
            for i in range(len(class_names))
        },
        "per_class_accuracy_calibrated": (
            {
                class_names[i]: (
                    round(100.0 * per_class_correct_cal[i] / per_class_total[i], 2)
                    if per_class_total[i] > 0
                    else 0
                )
                for i in range(len(class_names))
            }
            if log_prior is not None
            else None
        ),
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
            "answer_label_style": args.answer_label_style,
            "include_neighbor_labels": args.include_neighbor_labels,
            "neighbor_label_format": args.neighbor_label_format,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(args.log_file, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    logger.info("Results logged to %s", args.log_file)

    if args.dump_logits_path and "scores" in eval_out:
        import numpy as np
        os.makedirs(os.path.dirname(args.dump_logits_path), exist_ok=True)
        npz_payload = {
            "scores": eval_out["scores"],
            "cls_labels": eval_out["cls_labels"],
            "predictions_raw": np.asarray(eval_out["predictions"], dtype=np.int64),
            "predictions_cal": np.asarray(eval_out["predictions_calibrated"], dtype=np.int64),
            "class_names": np.asarray(class_names),
            "lora_path": np.asarray(args.lora_path or ""),
            "max_samples": np.asarray(args.max_samples),
            "seed": np.asarray(args.seed),
        }
        if log_prior is not None:
            npz_payload["log_prior"] = log_prior.detach().cpu().numpy()
        if "per_pos_logprobs" in eval_out:
            npz_payload["per_pos_logprobs"] = eval_out["per_pos_logprobs"]
        np.savez(args.dump_logits_path, **npz_payload)
        logger.info("Dumped logits to %s (scores shape=%s)", args.dump_logits_path, eval_out["scores"].shape)


if __name__ == "__main__":
    main()
