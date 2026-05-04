"""
TMDLMTrainer: Topology-Masked Diffusion Language Model Trainer.

Extends MDLMTrainer to support topology-conditioned attention masks for
Text-Attributed Graph (TAG) node classification.

Key difference from MDLMTrainer:
  - Accepts an additional "topology_mask" field [b, l, l] in the batch.
  - Combines the topology_mask with the standard padding attention_mask
    before the forward pass, so each node's tokens can only attend to
    tokens belonging to k-hop neighbors (as defined by the graph adjacency).
  - Loss is computed only over target-node tokens (masked positions in the
    target-node segment); neighbor tokens are never masked and not in the loss.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import transformers

from dllm.core.trainers.mdlm import MDLMTrainer, MDLMConfig
from dllm.pipelines.tmdlm.eval_utils import masked_forward_and_score


@dataclass
class TMDLMConfig(MDLMConfig):
    # Lambda weight for the auxiliary classification cross-entropy loss.
    cls_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for classification CE loss added to diffusion ELBO."},
    )
    # Hop boundaries for the multi-scale mask variant (TM-DLM-MS).
    # At timestep t > ms_threshold, use 1-hop mask; at t <= ms_threshold, use 2-hop.
    # Set to 0.0 to disable multi-scale (always use the mask provided in the batch).
    ms_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "Multi-scale threshold t*. When > 0, use topology_mask_1hop for t > t* "
                "and topology_mask_2hop for t <= t*. Requires both mask fields in batch."
            )
        },
    )


class TMDLMTrainer(MDLMTrainer):
    """
    Trainer for Topology-Masked Diffusion Language Models on text-attributed graphs.

    Batch fields (in addition to standard MDLMTrainer fields):
        input_ids           [b, l]      Tokenized sequence S_v (target + neighbor tokens)
        labels              [b, l]      Same as input_ids; -100 at neighbor/pad positions
        attention_mask      [b, l]      Standard padding mask (1 = valid, 0 = pad)
        topology_mask       [b, l, l]   Hard binary topology mask M_v; 1 where attention
                                        is allowed between token positions.
        topology_mask_1hop  [b, l, l]   (Optional) 1-hop mask for TM-DLM-MS coarse phase.
        topology_mask_2hop  [b, l, l]   (Optional) 2-hop mask for TM-DLM-MS refine phase.
        cls_labels          [b]         (Optional) Integer node class labels for aux CE loss.
        label_token_indices [b, C]      (Optional) Positions of [LABEL] tokens in sequence.
    """

    def __init__(
        self,
        args: TMDLMConfig,
        *pargs,
        eval_class_token_ids: torch.Tensor | None = None,
        eval_class_names: list[str] | None = None,
        eval_dataset_name: str | None = None,
        eval_prompt_format: str = "mc_digit",
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        self.cls_loss_weight = args.cls_loss_weight
        self.ms_threshold = args.ms_threshold
        # Eval-acc configuration. When eval_class_token_ids is provided,
        # `evaluation_loop` switches from stochastic ELBO to deterministic
        # answer-window masking + class scoring (matching eval_logit.py).
        self.eval_class_token_ids = eval_class_token_ids
        self.eval_class_names = eval_class_names
        self.eval_dataset_name = eval_dataset_name
        self.eval_prompt_format = eval_prompt_format

    def _build_attention_mask(
        self,
        attention_mask: torch.Tensor | None,  # [b, l]
        topology_mask: torch.Tensor | None,  # [b, l, l]
    ) -> torch.Tensor | None:
        """
        Combine padding attention mask with topology mask.

        The HuggingFace transformer models accept attention_mask as either:
          - [b, l]       (standard padding mask)
          - [b, 1, l, l] (additive 4D mask, already expanded for multi-head)

        We use the 4D additive form so that topology constraints are applied
        to every head simultaneously without modifying the model architecture.

        Convention: additive mask uses 0 for "attend" and -inf (large negative)
        for "do not attend", following HF's convention in
        `modeling_utils.get_extended_attention_mask`.
        """
        if topology_mask is None:
            return attention_mask  # fall back to standard padding mask

        # topology_mask: 1 = allowed, 0 = blocked  [b, l, l]
        # Convert to additive mask: 0 where allowed, -inf where blocked.
        additive = torch.zeros_like(topology_mask, dtype=torch.float)
        additive = additive.masked_fill(topology_mask == 0, float("-inf"))  # [b, l, l]

        # Also apply padding: tokens at padding positions cannot be attended to.
        if attention_mask is not None:
            # attention_mask [b, l] -> [b, 1, l]: columns that are padding
            pad_cols = (1 - attention_mask).unsqueeze(1).bool()  # [b, 1, l]
            additive = additive.masked_fill(pad_cols, float("-inf"))

        # Expand to [b, 1, l, l] for HF multi-head attention.
        return additive.unsqueeze(1)

    def _select_topology_mask(
        self,
        inputs: dict,
        t: torch.Tensor,  # [b]
    ) -> torch.Tensor | None:
        """
        Select the topology mask per sample based on timestep (TM-DLM-MS variant).

        If ms_threshold > 0 and both 1-hop and 2-hop masks are provided in the batch,
        samples with t > ms_threshold use the 1-hop mask (coarse denoising phase) and
        samples with t <= ms_threshold use the 2-hop mask (refinement phase).
        """
        if self.ms_threshold > 0:
            mask_1hop = inputs.get("topology_mask_1hop", None)
            mask_2hop = inputs.get("topology_mask_2hop", None)
            if mask_1hop is not None and mask_2hop is not None:
                # Per-sample selection: [b] boolean -> [b, 1, 1] for broadcasting
                use_1hop = (t > self.ms_threshold).view(-1, 1, 1)
                return torch.where(use_1hop, mask_1hop, mask_2hop)

        return inputs.get("topology_mask", None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to inject topology-conditioned attention mask.

        The core change is in step 3 (forward pass): instead of passing the
        standard padding attention_mask, we pass the combined 4D topology+padding
        mask so the transformer attends only within the k-hop neighborhood.
        """
        inputs = self._preprocess_inputs(inputs)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        padding_mask = inputs.get("attention_mask", None)
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )

        # === 2. Select topology mask (may depend on t for TM-DLM-MS) ===
        topology_mask = self._select_topology_mask(inputs, t)

        # === 3. Apply stochastic masking (only on target-node tokens) ===
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)
        masked_mask = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # === 4. Build combined attention mask (padding + topology) ===
        combined_mask = self._build_attention_mask(padding_mask, topology_mask)

        # === 5. Forward pass with topology-constrained attention ===
        forward_kwargs = dict(input_ids=noised_input_ids, attention_mask=combined_mask)
        if "position_ids" in inputs:
            forward_kwargs["position_ids"] = inputs["position_ids"]
        outputs = model(**forward_kwargs)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits  # [b, l, V]

        # === 6. Diffusion ELBO loss (target-node tokens only) ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        assert (input_ids[maskable_mask] == labels[maskable_mask]).all()

        import torch.nn.functional as F

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),  # [b, V, l]
            input_ids,
            reduction="none",
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=logits.dtype).detach(),
        )

        if self.loss_norm_type == "token":
            token_nll /= maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        loss = token_nll.sum()

        # === 7. Auxiliary classification loss on [LABEL] tokens (optional) ===
        cls_labels = inputs.get("cls_labels", None)
        label_token_indices = inputs.get("label_token_indices", None)
        if (
            cls_labels is not None
            and label_token_indices is not None
            and self.cls_loss_weight > 0
        ):
            # label_token_indices: [b, C] positions of the C label tokens
            # Gather logits at those positions and compute CE over class vocabulary
            label_logits = logits[
                torch.arange(b, device=logits.device).unsqueeze(1),
                label_token_indices,
            ]  # [b, C, V] -> take max over C=1 for single-token labels
            label_logits = label_logits.squeeze(1)  # [b, V] assuming C=1

            # Restrict logits to answer digit tokens ("0", "1", ..., "K-1")
            # so cls_labels (0..K-1) correctly index into the restricted set.
            num_classes = int(cls_labels.max().item()) + 1
            digit_token_ids = [
                self.processing_class.encode(str(i), add_special_tokens=False)[0]
                for i in range(num_classes)
            ]
            digit_ids_tensor = torch.tensor(digit_token_ids, device=label_logits.device)
            restricted_logits = label_logits[:, digit_ids_tensor]  # [b, K]
            cls_loss = F.cross_entropy(restricted_logits, cls_labels)
            loss = loss + self.cls_loss_weight * cls_loss

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        """Deterministic-mask classification accuracy as the eval metric.

        Falls back to the parent stochastic-ELBO loop when eval-acc kwargs
        are not configured. When configured, mirrors eval_logit.py: mask the
        answer window, single forward pass, score against candidate class
        token IDs, return acc and (1 - acc) as eval_loss for HF early-stop /
        best-checkpoint compatibility.
        """
        if self.eval_class_token_ids is None:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        from transformers.trainer_utils import EvalLoopOutput

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        device = next(model.parameters()).device
        cls_ids = self.eval_class_token_ids.to(device)
        tokenizer = self.processing_class

        use_topology_mask = bool(
            getattr(dataloader, "collate_fn", None) is not None
            and getattr(dataloader.collate_fn, "use_topology_mask", False)
        )

        correct = 0
        total = 0
        for batch in dataloader:
            preds, cls_labels = masked_forward_and_score(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                class_token_ids_tensor=cls_ids,
                use_topology_mask=use_topology_mask,
                dataset_name=self.eval_dataset_name,
                prompt_format=self.eval_prompt_format,
                class_names=self.eval_class_names,
            )
            correct += (preds == cls_labels).sum().item()
            total += cls_labels.numel()

        acc = float(correct) / max(total, 1)
        metrics = {
            f"{metric_key_prefix}_acc": acc,
            f"{metric_key_prefix}_loss": 1.0 - acc,
            f"{metric_key_prefix}_correct": correct,
            f"{metric_key_prefix}_total": total,
        }
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=total,
        )
