"""
Shared scoring utilities used by both eval_logit.py and TMDLMTrainer's
deterministic-mask eval-acc loop. Keeping the scoring logic in one place
prevents drift between offline eval and online (during-SFT) eval.

Two public entry points:

- score_classes_from_logits:  given pre-computed logits at the answer
  positions, return the predicted class index per sample. Used by both
  the offline eval_logit.py loop and the trainer's eval-time hook.

- masked_forward_and_score:    one-shot helper that, given a batch from
  GraphDataCollator and a model, applies deterministic answer masking,
  runs a single forward pass, and calls score_classes_from_logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _valid_class_token_lists(class_token_ids, pad_id):
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
    log_probs: torch.Tensor,
    label_starts: torch.Tensor,
    class_names: list[str],
    class_token_ids_tensor: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Hierarchical PubMed scorer.

    Treat the shared prefix "Diabetes Mellitus" as prompt context. First
    compare the next token for Experimental (",") against the Type branch
    ("Type"). If Type wins, compare the following token between Type 1 and
    Type 2.
    """
    device = log_probs.device
    token_lists = _valid_class_token_lists(class_token_ids_tensor.tolist(), pad_id)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    exp_idx = name_to_idx["Diabetes Mellitus, Experimental"]
    type1_idx = name_to_idx["Diabetes Mellitus Type 1"]
    type2_idx = name_to_idx["Diabetes Mellitus Type 2"]

    shared_len = _shared_prefix_len(token_lists)
    type_shared_len = _shared_prefix_len(
        [token_lists[type1_idx], token_lists[type2_idx]]
    )

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

    preds = torch.full(
        (log_probs.shape[0],), type1_idx, device=device, dtype=torch.long
    )
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


def score_classes_from_logits(
    logits: torch.Tensor,
    label_indices: torch.Tensor,
    class_token_ids_tensor: torch.Tensor,
    pad_id: int,
    *,
    dataset_name: str | None = None,
    prompt_format: str = "mc_digit",
    class_names: list[str] | None = None,
) -> torch.Tensor:
    """Return per-sample predicted class index.

    Args:
        logits: [b, l, V] full-vocab logits from a single forward pass.
        label_indices: [b, max_ans_len] absolute positions of the answer
            window in the input sequence.
        class_token_ids_tensor: [K] (single-token) or [K, max_ans_len]
            (multi-token) tensor of candidate class token IDs.
        pad_id: token id used as filler in class_token_ids_tensor; positions
            equal to pad_id are excluded from the per-class score.
        dataset_name, prompt_format, class_names: only consulted to route
            the special "pubmed + category_infill" hierarchical scorer.

    Returns:
        preds: [b] predicted class indices.
    """
    device = logits.device
    b = logits.shape[0]

    if class_token_ids_tensor.dim() == 1:
        # Single-token (mc_digit, max_answer_tokens=1): restricted argmax.
        label_logits = logits[
            torch.arange(b, device=device).unsqueeze(1),
            label_indices,
        ].squeeze(1)  # [b, V]
        restricted_logits = label_logits[:, class_token_ids_tensor]  # [b, K]
        return restricted_logits.argmax(dim=-1)

    # Multi-token: gather per-position log-probs and mean across valid slots.
    ans_len = label_indices.shape[1]
    log_probs = F.log_softmax(logits, dim=-1)  # [b, l, V]

    pos_expanded = label_indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    ans_log_probs = log_probs.gather(1, pos_expanded)  # [b, ans_len, V]

    if dataset_name == "pubmed" and prompt_format == "category_infill":
        assert class_names is not None
        return _predict_pubmed_hierarchical(
            log_probs=log_probs,
            label_starts=label_indices[:, 0],
            class_names=class_names,
            class_token_ids_tensor=class_token_ids_tensor,
            pad_id=pad_id,
        )

    K = class_token_ids_tensor.shape[0]
    cls_ids = class_token_ids_tensor.unsqueeze(0).expand(b, -1, -1)  # [b, K, ans_len]
    cls_valid = class_token_ids_tensor != pad_id  # [K, ans_len]

    scores = torch.zeros(b, K, device=device)
    for j in range(ans_len):
        pos_logprobs = ans_log_probs[:, j, :]  # [b, V]
        cls_tokens_j = cls_ids[:, :, j]  # [b, K]
        token_scores = pos_logprobs.gather(1, cls_tokens_j)  # [b, K]
        valid_j = cls_valid[:, j].unsqueeze(0).expand(b, -1)  # [b, K]
        scores += token_scores * valid_j.float()

    cls_lengths = cls_valid.sum(dim=1).float().clamp(min=1)  # [K]
    scores = scores / cls_lengths.unsqueeze(0)  # mean log-prob per class
    return scores.argmax(dim=-1)


@torch.no_grad()
def masked_forward_and_score(
    *,
    model,
    batch: dict[str, torch.Tensor],
    tokenizer,
    class_token_ids_tensor: torch.Tensor,
    use_topology_mask: bool,
    dataset_name: str | None = None,
    prompt_format: str = "mc_digit",
    class_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot deterministic-mask eval step.

    Mirrors the masking + forward + scoring done by
    examples/tmdlm/eval_logit.py:evaluate_layer0 so trainer-time and
    offline eval produce identical predictions for the same checkpoint.

    Returns: (preds [b], cls_labels [b]).
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    cls_labels = batch["cls_labels"].to(device)
    label_indices = batch["label_token_indices"].to(device)
    b, l = input_ids.shape

    masked_input = input_ids.clone()
    b_arange = torch.arange(b, device=device)
    attn1d = batch.get("attention_mask", None)
    if attn1d is not None:
        attn1d = attn1d.clone().to(device)
    for j in range(label_indices.shape[1]):
        masked_input[b_arange, label_indices[:, j]] = tokenizer.mask_token_id
        if attn1d is not None:
            attn1d[b_arange, label_indices[:, j]] = 1

    if use_topology_mask and "topology_mask" in batch:
        topo = batch["topology_mask"].to(device)
        additive = torch.zeros(b, l, l, device=device, dtype=model.dtype)
        additive = additive.masked_fill(topo == 0, float("-inf"))
        attn_mask = additive.unsqueeze(1)  # [b, 1, l, l]
    else:
        attn_mask = attn1d

    pos_ids = batch.get("position_ids", None)
    if pos_ids is not None:
        pos_ids = pos_ids.to(device)
    forward_kwargs = dict(input_ids=masked_input, attention_mask=attn_mask)
    if pos_ids is not None:
        forward_kwargs["position_ids"] = pos_ids
    outputs = model(**forward_kwargs)
    logits = outputs.logits

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    preds = score_classes_from_logits(
        logits=logits,
        label_indices=label_indices,
        class_token_ids_tensor=class_token_ids_tensor,
        pad_id=pad_id,
        dataset_name=dataset_name,
        prompt_format=prompt_format,
        class_names=class_names,
    )
    return preds, cls_labels
