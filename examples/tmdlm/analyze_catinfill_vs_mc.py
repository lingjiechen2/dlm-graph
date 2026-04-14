"""
Analyze failure cases: mc_digit vs catinfill+options on Cora target-only.

Compares per-sample predictions to find where catinfill fails but mc_digit succeeds.
"""

import json
import torch
import torch.nn.functional as F
import transformers
from collections import Counter, defaultdict

import dllm
from dllm.data.graph import load_tag_dataset, get_class_token_ids
from dllm.pipelines.tmdlm.utils import GraphDataCollator

logger = dllm.utils.get_default_logger(__name__)


@torch.no_grad()
def get_predictions(
    model, tokenizer, dataset, class_token_ids, batch_size, max_answer_tokens
):
    """Run evaluation and return per-sample predictions + scores."""
    device = next(model.parameters()).device

    if max_answer_tokens == 1:
        class_ids_tensor = torch.tensor(class_token_ids, device=device)
    else:
        class_ids_tensor = torch.tensor(
            class_token_ids, device=device, dtype=torch.long
        )

    collator = GraphDataCollator(tokenizer=tokenizer, padding=True, return_tensors="pt")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    all_preds = []
    all_scores = []  # per-class scores for each sample

    for batch in dataloader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        label_indices = batch["label_token_indices"]
        b, l = input_ids.shape

        # Mask answer positions
        masked_input = input_ids.clone()
        if max_answer_tokens == 1:
            maskable = labels != -100
            masked_input[maskable] = tokenizer.mask_token_id
        else:
            maskable = labels != -100
            masked_input[maskable] = tokenizer.mask_token_id
            for j in range(label_indices.shape[1]):
                masked_input[torch.arange(b, device=device), label_indices[:, j]] = (
                    tokenizer.mask_token_id
                )

        attn_mask = batch.get("attention_mask", None)
        outputs = model(input_ids=masked_input, attention_mask=attn_mask)
        logits = outputs.logits

        if max_answer_tokens == 1:
            label_logits = logits[
                torch.arange(b, device=device).unsqueeze(1), label_indices
            ].squeeze(1)
            restricted = label_logits[:, class_ids_tensor]
            preds = restricted.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(restricted.cpu().tolist())
        else:
            ans_len = label_indices.shape[1]
            log_probs = F.log_softmax(logits, dim=-1)
            pos_expanded = label_indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
            ans_log_probs = log_probs.gather(1, pos_expanded)

            K = class_ids_tensor.shape[0]
            cls_ids = class_ids_tensor.unsqueeze(0).expand(b, -1, -1)

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            cls_valid = class_ids_tensor != pad_id

            scores = torch.zeros(b, K, device=device)
            for j in range(ans_len):
                pos_logprobs = ans_log_probs[:, j, :]
                cls_tokens_j = cls_ids[:, :, j]
                token_scores = pos_logprobs.gather(1, cls_tokens_j)
                valid_j = cls_valid[:, j].unsqueeze(0).expand(b, -1)
                scores += token_scores * valid_j.float()

            cls_lengths = cls_valid.sum(dim=1).float().clamp(min=1)
            scores = scores / cls_lengths.unsqueeze(0)
            preds = scores.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(scores.cpu().tolist())

    return all_preds, all_scores


def main():
    # Load model once
    model_path = dllm.utils.resolve_with_base_env(
        "GSAI-ML/LLaDA-8B-Instruct", "BASE_MODELS_DIR"
    )
    model_args = dllm.utils.ModelArguments(model_name_or_path=model_path)
    model = dllm.utils.get_model(model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args)
    model.eval()

    class_names_mc, class_ids_mc = get_class_token_ids(
        "cora", tokenizer, max_answer_tokens=1, prompt_format="mc_digit"
    )
    class_names_ci, class_ids_ci = get_class_token_ids(
        "cora", tokenizer, max_answer_tokens=1, prompt_format="category_infill"
    )
    # Auto-compute max_answer_tokens for catinfill
    max_ans_ci = len(class_ids_ci[0])

    logger.info("MC class IDs: %s", class_ids_mc)
    logger.info("CI class IDs (max_ans=%d): %s", max_ans_ci, class_ids_ci)

    # Load datasets with both formats
    logger.info("Loading mc_digit dataset...")
    ds_mc = load_tag_dataset(
        "cora",
        tokenizer=tokenizer,
        split="test",
        max_seq_len=2048,
        max_neighbors_per_hop=10,
        max_hops=0,
        seed=42,
        max_answer_tokens=1,
        prompt_format="mc_digit",
    )
    logger.info("Loading catinfill+options dataset...")
    ds_ci = load_tag_dataset(
        "cora",
        tokenizer=tokenizer,
        split="test",
        max_seq_len=2048,
        max_neighbors_per_hop=10,
        max_hops=0,
        seed=42,
        max_answer_tokens=max_ans_ci,
        prompt_format="category_infill",
    )

    # Run predictions
    logger.info("Running mc_digit predictions...")
    preds_mc, scores_mc = get_predictions(
        model, tokenizer, ds_mc, class_ids_mc, batch_size=8, max_answer_tokens=1
    )
    logger.info("Running catinfill+options predictions...")
    preds_ci, scores_ci = get_predictions(
        model,
        tokenizer,
        ds_ci,
        class_ids_ci,
        batch_size=8,
        max_answer_tokens=max_ans_ci,
    )

    # Collect ground truth labels
    gt_labels = [ds_mc[i]["cls_label"] for i in range(len(ds_mc))]

    # Analyze
    n = len(gt_labels)
    mc_correct = [preds_mc[i] == gt_labels[i] for i in range(n)]
    ci_correct = [preds_ci[i] == gt_labels[i] for i in range(n)]

    mc_right_ci_wrong = []  # mc correct, catinfill wrong
    ci_right_mc_wrong = []  # catinfill correct, mc wrong
    both_right = []
    both_wrong = []

    for i in range(n):
        if mc_correct[i] and not ci_correct[i]:
            mc_right_ci_wrong.append(i)
        elif ci_correct[i] and not mc_correct[i]:
            ci_right_mc_wrong.append(i)
        elif mc_correct[i] and ci_correct[i]:
            both_right.append(i)
        else:
            both_wrong.append(i)

    logger.info("=" * 70)
    logger.info("COMPARISON: mc_digit vs catinfill+options (Cora target-only)")
    logger.info("=" * 70)
    logger.info("Total samples: %d", n)
    logger.info(
        "mc_digit correct: %d (%.1f%%)", sum(mc_correct), 100.0 * sum(mc_correct) / n
    )
    logger.info(
        "catinfill correct: %d (%.1f%%)", sum(ci_correct), 100.0 * sum(ci_correct) / n
    )
    logger.info("")
    logger.info("Both correct:           %d", len(both_right))
    logger.info(
        "mc correct, ci wrong:   %d  <-- catinfill failures", len(mc_right_ci_wrong)
    )
    logger.info(
        "ci correct, mc wrong:   %d  <-- catinfill wins", len(ci_right_mc_wrong)
    )
    logger.info("Both wrong:             %d", len(both_wrong))

    # Per-class breakdown of mc_right_ci_wrong
    logger.info("")
    logger.info("--- mc correct, catinfill wrong: per GT class ---")
    gt_class_counter = Counter(gt_labels[i] for i in mc_right_ci_wrong)
    pred_ci_counter = Counter(preds_ci[i] for i in mc_right_ci_wrong)
    for cls_id in sorted(gt_class_counter.keys()):
        logger.info(
            "  GT class %d (%s): %d samples",
            cls_id,
            class_names_mc[cls_id],
            gt_class_counter[cls_id],
        )
    logger.info("")
    logger.info("  catinfill predicted instead:")
    for cls_id in sorted(pred_ci_counter.keys()):
        logger.info(
            "    -> class %d (%s): %d times",
            cls_id,
            class_names_mc[cls_id],
            pred_ci_counter[cls_id],
        )

    # Confusion: GT -> catinfill prediction for mc_right_ci_wrong cases
    logger.info("")
    logger.info(
        "--- Confusion: GT class -> catinfill prediction (mc correct, ci wrong) ---"
    )
    confusion = defaultdict(Counter)
    for i in mc_right_ci_wrong:
        gt = gt_labels[i]
        ci_pred = preds_ci[i]
        confusion[gt][ci_pred] += 1
    for gt_cls in sorted(confusion.keys()):
        preds_str = ", ".join(
            f"{class_names_mc[p]}({c})" for p, c in confusion[gt_cls].most_common()
        )
        logger.info("  %s -> %s", class_names_mc[gt_cls], preds_str)

    # Per-class breakdown of ci_right_mc_wrong
    logger.info("")
    logger.info("--- catinfill correct, mc wrong: per GT class ---")
    gt_class_counter2 = Counter(gt_labels[i] for i in ci_right_mc_wrong)
    for cls_id in sorted(gt_class_counter2.keys()):
        logger.info(
            "  GT class %d (%s): %d samples",
            cls_id,
            class_names_mc[cls_id],
            gt_class_counter2[cls_id],
        )

    # Score analysis: for mc_right_ci_wrong, show score margins
    logger.info("")
    logger.info("--- Score margin analysis (mc correct, ci wrong) ---")
    margins = []
    for i in mc_right_ci_wrong:
        gt = gt_labels[i]
        ci_scores = scores_ci[i]
        gt_score = ci_scores[gt]
        pred_score = max(ci_scores)
        margin = pred_score - gt_score
        margins.append(margin)

    margins_sorted = sorted(margins)
    logger.info("  Margin (wrong_class - correct_class) in catinfill log-prob:")
    logger.info("    min: %.4f", margins_sorted[0])
    logger.info("    25%%: %.4f", margins_sorted[len(margins_sorted) // 4])
    logger.info("    50%%: %.4f", margins_sorted[len(margins_sorted) // 2])
    logger.info("    75%%: %.4f", margins_sorted[3 * len(margins_sorted) // 4])
    logger.info("    max: %.4f", margins_sorted[-1])

    # Show a few example failures
    logger.info("")
    logger.info("--- Example failures (mc correct, ci wrong, sorted by margin) ---")
    indexed_margins = [
        (margins[j], mc_right_ci_wrong[j]) for j in range(len(mc_right_ci_wrong))
    ]
    # Show smallest margins first (most "borderline" cases)
    indexed_margins.sort()
    for margin, idx in indexed_margins[:10]:
        gt = gt_labels[idx]
        mc_pred = preds_mc[idx]
        ci_pred = preds_ci[idx]
        text = tokenizer.decode(ds_mc[idx]["input_ids"][:80])
        ci_scores_str = " ".join(
            f"{class_names_mc[k][:6]}={scores_ci[idx][k]:.2f}"
            for k in range(len(class_names_mc))
        )
        logger.info(
            "  [%d] GT=%s mc=%s ci=%s margin=%.3f | %s",
            idx,
            class_names_mc[gt][:12],
            class_names_mc[mc_pred][:12],
            class_names_mc[ci_pred][:12],
            margin,
            ci_scores_str,
        )
        logger.info("       text: %s...", text[:120])

    # Also show largest margins (most confident wrong predictions)
    logger.info("")
    logger.info("--- Most confident wrong predictions (largest margin) ---")
    for margin, idx in indexed_margins[-5:]:
        gt = gt_labels[idx]
        ci_pred = preds_ci[idx]
        ci_scores_str = " ".join(
            f"{class_names_mc[k][:6]}={scores_ci[idx][k]:.2f}"
            for k in range(len(class_names_mc))
        )
        logger.info(
            "  [%d] GT=%s ci=%s margin=%.3f | %s",
            idx,
            class_names_mc[gt][:12],
            class_names_mc[ci_pred][:12],
            margin,
            ci_scores_str,
        )

    # Token-length effect analysis
    logger.info("")
    logger.info("--- Token-length effect ---")
    token_lens = {
        i: sum(1 for t in class_ids_ci[i] if t != tokenizer.pad_token_id)
        for i in range(len(class_names_mc))
    }
    for cls_id in range(len(class_names_mc)):
        total = sum(1 for g in gt_labels if g == cls_id)
        mc_c = sum(1 for j in range(n) if gt_labels[j] == cls_id and mc_correct[j])
        ci_c = sum(1 for j in range(n) if gt_labels[j] == cls_id and ci_correct[j])
        logger.info(
            "  %s (%d tokens): mc=%d/%d (%.1f%%) ci=%d/%d (%.1f%%) Δ=%+.1f%%",
            class_names_mc[cls_id],
            token_lens[cls_id],
            mc_c,
            total,
            100 * mc_c / total if total else 0,
            ci_c,
            total,
            100 * ci_c / total if total else 0,
            100 * (ci_c - mc_c) / total if total else 0,
        )


if __name__ == "__main__":
    main()
