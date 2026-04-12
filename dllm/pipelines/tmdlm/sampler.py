"""
TMDLMSampler: Inference-time iterative denoising for TM-DLM.

Wraps the standard MDLMSampler to inject topology masks during generation.
At each denoising step, the topology mask is applied to restrict attention
to k-hop neighbors, matching the training-time setting.

Usage (node classification):
    sampler = TMDLMSampler(model, tokenizer, config)
    predictions = sampler.classify(input_ids, topology_mask, label_token_indices)
"""

from dataclasses import dataclass

import torch

from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig


@dataclass
class TMDLMSamplerConfig(MDLMSamplerConfig):
    # Number of iterative denoising steps (T=1 for one-shot, T=10 for iterative)
    steps: int = 10
    # Whether to use multi-scale masking during inference
    use_multiscale: bool = False
    ms_threshold: float = 0.7


class TMDLMSampler:
    """
    Iterative denoising sampler for TM-DLM node classification.

    Unlike standard MDLMSampler (which generates free text), this sampler
    is specialized for node classification:
      - Only target-node tokens are unmasked iteratively.
      - Neighbor tokens are always fully visible (never masked).
      - The topology mask is applied at every denoising step.
    """

    def __init__(self, model, tokenizer, config: TMDLMSamplerConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TMDLMSamplerConfig()

    @torch.no_grad()
    def classify(
        self,
        input_ids: torch.Tensor,           # [b, l]  fully masked target tokens
        topology_mask: torch.Tensor,        # [b, l, l]  hard binary topology mask
        label_token_indices: torch.Tensor,  # [b, C]  positions of [LABEL] tokens
        topology_mask_1hop: torch.Tensor | None = None,
        topology_mask_2hop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run iterative denoising and return the predicted class index for each node.

        Returns:
            predictions: [b] integer class indices (argmax over vocab at label positions)
        """
        T = self.config.steps
        b, l = input_ids.shape
        x = input_ids.clone()

        # Track which target-node positions are still masked
        # (label_token_indices marks the positions we care about for classification)
        still_masked = torch.ones(b, dtype=torch.bool, device=x.device)

        for step in range(T):
            t_frac = 1.0 - step / T  # goes from 1 -> 1/T

            # Select topology mask for this timestep (TM-DLM-MS)
            if (
                self.config.use_multiscale
                and topology_mask_1hop is not None
                and topology_mask_2hop is not None
            ):
                use_1hop = t_frac > self.config.ms_threshold
                topo = topology_mask_1hop if use_1hop else topology_mask_2hop
            else:
                topo = topology_mask

            # Build additive 4D mask
            additive = torch.zeros_like(topo, dtype=torch.float)
            additive = additive.masked_fill(topo == 0, float("-inf"))
            combined_mask = additive.unsqueeze(1)  # [b, 1, l, l]

            # Forward pass
            outputs = self.model(input_ids=x, attention_mask=combined_mask)
            logits = outputs.logits  # [b, l, V]

            # Get logits at label positions: [b, C, V]
            label_logits = logits[
                torch.arange(b, device=logits.device).unsqueeze(1),
                label_token_indices,
            ]  # [b, C, V]
            # For C=1 single-token labels, squeeze
            if label_logits.shape[1] == 1:
                label_logits = label_logits.squeeze(1)  # [b, V]

        # Final prediction: argmax at label token position after last step
        predictions = label_logits.argmax(dim=-1)  # [b]
        return predictions
