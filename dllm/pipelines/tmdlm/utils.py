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

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        b = len(features)

        # --- Pad input_ids and labels ---
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((b, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((b, max_len), self.label_pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(b, max_len, dtype=torch.long)

        for i, f in enumerate(features):
            seq_len = len(f["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1

        # --- Build topology masks ---
        topology_mask = self._build_topology_mask(features, max_len, hop_limit=None)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "topology_mask": topology_mask,
        }

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
            batch["label_token_indices"] = torch.tensor(
                [[f["label_token_pos"]] for f in features], dtype=torch.long
            )  # [b, 1]

        return batch

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
            if not spans:
                # Fallback: full attention
                seq_len = len(f["input_ids"])
                mask[i, :seq_len, :seq_len] = 1
                continue

            target_start, target_end = spans[0]  # (start, end) exclusive

            for node_idx, (start, end) in enumerate(spans):
                hop = f["node_hops"][node_idx] if "node_hops" in f else node_idx
                if hop_limit is not None and hop > hop_limit:
                    continue

                if node_idx == 0:
                    # Target node: attend to all tokens in the sequence (within valid spans)
                    for other_start, other_end in spans:
                        other_hop = f["node_hops"][spans.index((other_start, other_end))] \
                            if "node_hops" in f else spans.index((other_start, other_end))
                        if hop_limit is None or other_hop <= hop_limit:
                            mask[i, start:end, other_start:other_end] = 1
                else:
                    # Neighbor node: attend to self + target node tokens only
                    mask[i, start:end, start:end] = 1                  # self
                    mask[i, start:end, target_start:target_end] = 1    # target

        return mask
