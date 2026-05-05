"""
Graph data utilities for TM-DLM.

GraphDataCollator: collates per-node samples into a padded batch and
builds the topology attention mask M_v from stored adjacency information.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class GraphDataCollator:
    """
    Data collator for TM-DLM node classification.

    Each sample in the dataset is expected to have:
        input_ids       list[int]       Tokenized S_v sequence
        labels          list[int]       Same as input_ids; -100 at neighbor positions
        node_spans      list[(int,int)] Token span (start, end) per node in the sequence.
                                        node_spans[0] = target node, rest = neighbors.
        node_hops       list[int]       Hop distance for each node (0=target, 1=1-hop, ...)
        cls_label       int             Integer class label for this node
        label_token_pos int             Index of the [LABEL] token in input_ids

    Optionally, if 1-hop and 2-hop masks are needed separately (for TM-DLM-MS),
    node_spans_1hop and node_spans_2hop can be provided.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    return_tensors: str = "pt"
    label_pad_token_id: int = -100
    position_id_type: str = "sequential"  # "sequential" (default) or "topological"
    use_topology_mask: bool = True  # False → fall back to dense padding-only attention

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        b = len(features)

        # --- Pad input_ids and labels ---
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full(
            (b, max_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        labels = torch.full((b, max_len), self.label_pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(b, max_len, dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = len(f["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            # Zero attention on the reserved-tail pad span in the answer slot
            # (positions filled with pad_token_id by build_node_sample_category).
            # This prevents the model from "seeing" those positions during
            # training, which would otherwise leak GT class length via the
            # trailing-pad count. Empty when answer fits exactly.
            ap_start = f.get("answer_pad_start")
            ap_end = f.get("answer_pad_end")
            if ap_start is not None and ap_end is not None and ap_end > ap_start:
                attention_mask[i, ap_start:ap_end] = 0

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # --- Build topology masks (skip when disabled to use dense attention) ---
        if self.use_topology_mask:
            batch["topology_mask"] = self._build_topology_mask(
                features, max_len, hop_limit=None
            )

        # --- Build topological position IDs (optional) ---
        if self.position_id_type == "topological":
            batch["position_ids"] = self._build_topological_position_ids(
                features, max_len
            )

        # Optional: 1-hop and 2-hop masks for TM-DLM-MS
        if all("node_spans_1hop" in f for f in features):
            batch["topology_mask_1hop"] = self._build_topology_mask(
                features, max_len, hop_limit=1, spans_key="node_spans_1hop"
            )
        if all("node_spans_2hop" in f for f in features):
            batch["topology_mask_2hop"] = self._build_topology_mask(
                features, max_len, hop_limit=2, spans_key="node_spans_2hop"
            )

        # Optional: classification labels and label token positions
        if all("cls_label" in f for f in features):
            batch["cls_labels"] = torch.tensor(
                [f["cls_label"] for f in features], dtype=torch.long
            )
        if all("label_token_pos" in f for f in features):
            max_ans_len = max(f.get("answer_len", 1) for f in features)
            label_indices = []
            for f in features:
                start = f["label_token_pos"]
                # Always use consecutive positions from start, even for short answers.
                # The input_ids already has max_answer_tokens positions allocated
                # (real tokens + pad tokens), so start..start+max_ans_len-1 are valid.
                positions = list(range(start, start + max_ans_len))
                label_indices.append(positions)
            batch["label_token_indices"] = torch.tensor(
                label_indices, dtype=torch.long
            )  # [b, max_ans_len]
            batch["answer_lens"] = torch.tensor(
                [f.get("answer_len", 1) for f in features], dtype=torch.long
            )  # [b]

        return batch

    def _build_topological_position_ids(
        self,
        features: list[dict],
        max_len: int,
    ) -> torch.Tensor:
        """
        Build position IDs where each node's tokens restart from 0.

        Instead of sequential [0, 1, ..., L-1] across the entire concatenated
        sequence, each node (target and every neighbor) gets its own position
        IDs starting from 0.  This removes the concatenation-order bias in
        RoPE: all neighbors are positionally "equidistant" from the target.

        Tokens outside any node span (e.g. padding) get position 0.
        """
        b = len(features)
        position_ids = torch.zeros(b, max_len, dtype=torch.long)

        for i, f in enumerate(features):
            spans = f.get("node_spans", [])
            if not spans:
                # Fallback: sequential positions
                seq_len = len(f["input_ids"])
                position_ids[i, :seq_len] = torch.arange(seq_len)
                continue

            for start, end in spans:
                node_len = end - start
                position_ids[i, start:end] = torch.arange(node_len)

        return position_ids

    def _build_topology_mask(
        self,
        features: list[dict],
        max_len: int,
        hop_limit: int | None,
        spans_key: str = "node_spans",
    ) -> torch.Tensor:
        """
        Build the binary topology mask M_v [b, max_len, max_len].

        M_v[i, j] = 1 iff:
          - j is in the target node's span (tokens from target can attend to everyone)
          - OR i and j belong to the same node
          - OR i is a neighbor token that can attend to the target node's tokens

        Topology-mask semantics (from method spec):
          - Target node tokens: attend to self + all neighbor tokens
          - Neighbor tokens: attend to self + target node tokens only
          - No cross-neighbor attention
        """
        b = len(features)
        mask = torch.zeros(b, max_len, max_len, dtype=torch.long)

        for i, f in enumerate(features):
            spans = f.get(spans_key, f.get("node_spans", []))
            roles = f.get("node_roles")
            if not spans:
                # Fallback: full attention
                seq_len = len(f["input_ids"])
                mask[i, :seq_len, :seq_len] = 1
                continue

            if roles is not None:
                self._fill_lp_mask(mask[i], spans, roles)
                continue

            hops = f.get("node_hops", list(range(len(spans))))
            target_start, target_end = spans[0][0], spans[0][1]

            for node_idx in range(len(spans)):
                start, end = spans[node_idx][0], spans[node_idx][1]
                hop = hops[node_idx]
                if hop_limit is not None and hop > hop_limit:
                    continue

                if node_idx == 0:
                    # Target node: attend to all tokens in the sequence (within valid spans)
                    for other_idx in range(len(spans)):
                        other_hop = hops[other_idx]
                        if hop_limit is None or other_hop <= hop_limit:
                            os, oe = spans[other_idx][0], spans[other_idx][1]
                            mask[i, start:end, os:oe] = 1
                else:
                    # Neighbor node: attend to self + target node tokens only
                    mask[i, start:end, start:end] = 1  # self
                    mask[i, start:end, target_start:target_end] = 1  # target

        return mask

    @staticmethod
    def _fill_lp_mask(
        sample_mask: torch.Tensor,
        spans: list,
        roles: list,
    ) -> None:
        """Populate one [L, L] mask slice with LP crossed-star semantics.

        Roles ∈ {0=TARGET, 1=NBR_U, 2=NBR_V, 3=QUESTION}. Allowed (row -> col):
            TARGET   -> TARGET, NBR_U, NBR_V, QUESTION   (everything)
            NBR_U    -> TARGET, self                      (no NBR_V, no Q)
            NBR_V    -> TARGET, self                      (no NBR_U, no Q)
            QUESTION -> TARGET, self
        Same per-token cost as the NC star (centres do the heavy work).
        """
        target_spans = [spans[k] for k, r in enumerate(roles) if int(r) == 0]
        u_nbr_spans = [spans[k] for k, r in enumerate(roles) if int(r) == 1]
        v_nbr_spans = [spans[k] for k, r in enumerate(roles) if int(r) == 2]
        q_spans = [spans[k] for k, r in enumerate(roles) if int(r) == 3]

        def _fill(rows, cols):
            for rs, re in rows:
                for cs, ce in cols:
                    sample_mask[rs:re, cs:ce] = 1

        _fill(target_spans, target_spans + u_nbr_spans + v_nbr_spans + q_spans)
        _fill(u_nbr_spans, target_spans)
        for s in u_nbr_spans:
            sample_mask[s[0]:s[1], s[0]:s[1]] = 1
        _fill(v_nbr_spans, target_spans)
        for s in v_nbr_spans:
            sample_mask[s[0]:s[1], s[0]:s[1]] = 1
        _fill(q_spans, target_spans)
        for s in q_spans:
            sample_mask[s[0]:s[1], s[0]:s[1]] = 1
