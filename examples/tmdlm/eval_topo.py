"""
TM-DLM: Node Classification Evaluation.

Evaluates a fine-tuned TM-DLM checkpoint on the test split of a TAG dataset.
Reports accuracy aligned with LLaGA's evaluation protocol.

Usage
-----
    python examples/tmdlm/eval.py \
        --model_name_or_path .models/tmdlm-llada-8b-cora/checkpoint-final \
        --dataset_name cora \
        --denoising_steps 10

Layer 0 (train-free diagnostic):
    python examples/tmdlm/eval.py \
        --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
        --dataset_name cora \
        --train_free True \
        --denoising_steps 1
"""

from dataclasses import dataclass, field

import torch
import transformers
from tqdm import tqdm

import dllm
from dllm.pipelines.tmdlm import TMDLMSampler, TMDLMSamplerConfig
from dllm.pipelines.tmdlm.utils import GraphDataCollator
from dllm.data.graph import load_tag_dataset, DATASET_CONFIGS

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class EvalArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = ".models/tmdlm/checkpoint-final"
    dataset_name: str = field(
        default="cora",
        metadata={"help": "TAG dataset: cora | pubmed | ogbn-arxiv | ogbn-products"},
    )
    denoising_steps: int = field(
        default=10,
        metadata={"help": "Number of iterative denoising steps (T=1 for one-shot)"},
    )
    train_free: bool = field(
        default=False,
        metadata={
            "help": "Skip fine-tuning; use frozen pretrained model (Layer 0 eval)"
        },
    )
    batch_size: int = field(default=16)
    max_seq_len: int = field(default=2048)
    max_neighbors_per_hop: int = field(default=10)
    max_hops: int = field(default=2)
    use_multiscale: bool = field(
        default=False,
        metadata={
            "help": "Enable TM-DLM-MS multi-scale topology mask during inference"
        },
    )
    prompt_layout: str = field(
        default="target_first",
        metadata={
            "help": "Prompt layout: target_first | neighbor_first",
            "choices": ["target_first", "neighbor_first"],
        },
    )


def evaluate():
    parser = transformers.HfArgumentParser(EvalArguments)
    (args,) = parser.parse_args_into_dataclasses()

    # --- Model & tokenizer ---
    model = dllm.utils.get_model(args)
    tokenizer = dllm.utils.get_tokenizer(args)
    model.eval()

    # --- Dataset ---
    logger.info("Loading test split of %s...", args.dataset_name)
    test_dataset = load_tag_dataset(
        args.dataset_name,
        tokenizer=tokenizer,
        split="test",
        max_seq_len=args.max_seq_len,
        max_neighbors_per_hop=args.max_neighbors_per_hop,
        max_hops=args.max_hops,
        prompt_layout=args.prompt_layout,
    )

    collator = GraphDataCollator(tokenizer=tokenizer, padding=True, return_tensors="pt")
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # --- Sampler ---
    sampler_config = TMDLMSamplerConfig(
        steps=args.denoising_steps,
        use_multiscale=args.use_multiscale,
    )
    sampler = TMDLMSampler(model, tokenizer, sampler_config)

    # --- Build label token id lookup for this dataset ---
    num_classes = DATASET_CONFIGS[args.dataset_name]["num_classes"]

    # --- Evaluation loop ---
    device = next(model.parameters()).device
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Evaluating {args.dataset_name}"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        preds = sampler.classify(
            input_ids=batch["input_ids"],
            topology_mask=batch["topology_mask"],
            label_token_indices=batch["label_token_indices"],
            topology_mask_1hop=batch.get("topology_mask_1hop"),
            topology_mask_2hop=batch.get("topology_mask_2hop"),
        )  # [b] token ids of predicted class names

        # Map predicted token ids back to class indices
        # (class name tokens are stored in dataset metadata)
        cls_labels = batch["cls_labels"]  # [b] ground-truth class indices

        # TODO: map preds (vocab token ids) -> class indices via class_name_token_ids lookup
        # For now, compare directly if label_token_ids == cls integer labels (placeholder)
        correct += (preds == cls_labels).sum().item()
        total += cls_labels.shape[0]

    accuracy = 100.0 * correct / total
    logger.info(
        "Accuracy on %s test set: %.2f%%  (%d/%d)",
        args.dataset_name,
        accuracy,
        correct,
        total,
    )
    return accuracy


if __name__ == "__main__":
    evaluate()
