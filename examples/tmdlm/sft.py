"""
TM-DLM: Supervised Fine-Tuning for Node Classification on Text-Attributed Graphs.

Usage
-----
Single GPU (testing):
    python examples/tmdlm/sft.py \
        --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
        --dataset_name cora \
        --output_dir .models/tmdlm-llada-8b-cora

Multi-GPU (DDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml \
        examples/tmdlm/sft.py \
        --dataset_name ogbn-arxiv \
        --output_dir .models/tmdlm-llada-8b-arxiv
"""

import os
from dataclasses import dataclass, field
from datetime import timedelta

# Pre-init the process group with a long timeout BEFORE any code touches
# accelerate.PartialState() — accelerate's first init uses the default 10 min
# TCPStore timeout, which is locked once the PG is created and can't be raised
# by HF Trainer's --ddp_timeout later. Large LP datasets (e.g. ogbn-arxiv 90k
# pairs, ~10-15 min build per rank) need a longer window before the first NCCL
# collective. Only fires under DDP.
if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    import torch.distributed as _dist
    if not _dist.is_initialized():
        _dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=int(os.environ.get("DDP_INIT_TIMEOUT", "3600"))),
        )

import accelerate
import transformers

import dllm
from dllm.pipelines import tmdlm
from dllm.data.graph import load_tag_dataset, load_lp_dataset

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"


@dataclass
class DataArguments:
    task: str = field(
        default="nc",
        metadata={
            "help": (
                "Task type: 'nc' (node classification, default) or "
                "'lp' (link prediction). 'lp' v1 supports cora only and "
                "ignores prompt_format / answer_label_style / neighbor-label flags."
            )
        },
    )
    lp_neg_ratio: int = field(
        default=1,
        metadata={"help": "Negative samples per positive for link prediction (default 1:1)."},
    )
    lp_hard_neg_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "Fraction of random negatives to replace with 2-hop structural hard negatives "
                "(nodes sharing a common neighbour with u but no direct edge). "
                "0.0 = all random (default), 0.5 = half hard half random, 1.0 = all hard."
            )
        },
    )
    lp_use_llaga_split: bool = field(
        default=False,
        metadata={
            "help": (
                "Use LLaGA's official edge_sampled_2_10_only_{train,test}.jsonl split "
                "instead of our random seed-42 split. Fixes train/test leakage when "
                "comparing against LLaGA baselines."
            )
        },
    )
    dataset_name: str = field(
        default="cora",
        metadata={
            "help": (
                "TAG dataset: cora | pubmed | ogbn-arxiv | ogbn-products. "
                "Pass a comma-separated list (e.g. 'cora,pubmed') to merge "
                "multiple datasets into a single shuffled training set."
            )
        },
    )
    max_seq_len: int = field(
        default=2048,
        metadata={"help": "Maximum token sequence length for S_v"},
    )
    max_neighbors_per_hop: int = field(
        default=10,
        metadata={"help": "Number of neighbors sampled per hop (LLaGA default: 10)"},
    )
    max_hops: int = field(
        default=2,
        metadata={"help": "Number of hops for neighborhood context (1 or 2)"},
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes for dataset preprocessing"},
    )
    mask_target_text: bool = field(
        default=False,
        metadata={
            "help": "Mask all target body tokens (not just answer digit) for denser training signal"
        },
    )
    position_id_type: str = field(
        default="sequential",
        metadata={
            "help": "Position ID scheme: 'sequential' (default [0..L-1]) or 'topological' (per-node reset)"
        },
    )
    prompt_format: str = field(
        default="mc_digit",
        metadata={
            "help": (
                "Prompt format: 'mc_digit' (digit answer) | 'category_infill' "
                "(class-name infill with eos padding) | 'nd_describe' "
                "(LLaGA-style node description, center text hidden) | "
                "'nda_describe' (same plus abstract in answer)"
            )
        },
    )
    answer_label_style: str = field(
        default="digit0",
        metadata={
            "help": "Answer label style for mc_digit prompts: digit0 | number1 | letter"
        },
    )
    max_answer_tokens: int = field(
        default=1,
        metadata={
            "help": "Answer token budget. Use 1 for mc_digit; 4-6 for category_infill (real answer supervised, reserved tail kept as unsupervised mask placeholders)"
        },
    )
    include_neighbor_labels: bool = field(
        default=False,
        metadata={
            "help": "If True, inject neighbor class names into the prompt and supervise those class-name tokens."
        },
    )
    neighbor_label_format: str = field(
        default="bracket",
        metadata={
            "help": "Neighbor label phrasing when include_neighbor_labels=True: bracket|paren|sentence|colon"
        },
    )
    mask_neighbor_labels: bool = field(
        default=False,
        metadata={
            "help": "If True, replace neighbor class-name tokens with [MASK] in input_ids and train on them jointly with the target answer."
        },
    )
    use_topology_mask: bool = field(
        default=True,
        metadata={
            "help": "Apply star-topology attention mask restricting neighbor-target attention"
        },
    )
    balance_merged: bool = field(
        default=False,
        metadata={
            "help": "Legacy flag: equivalent to --resample_strategy balance_datasets when merging."
        },
    )
    resample_strategy: str = field(
        default="none",
        metadata={
            "help": "Resampling strategy applied to per-dataset sample lists: 'none' | 'balance_datasets' (downsample each merged dataset to min count) | 'balance_classes' (per-dataset, resample each class to median count) | 'boost' (oversample specified classes via --boost_spec)"
        },
    )
    boost_spec: str = field(
        default="",
        metadata={
            "help": "Comma-separated 'dataset:class_name:factor' entries used when resample_strategy=boost. E.g. 'cora:Theory:2,cora:Rule Learning:3'."
        },
    )
    max_train_samples: int = field(
        default=0,
        metadata={
            "help": "If > 0, subsample the train split to this many samples (deterministic, takes the first N split_ids after seed shuffling). Useful for large datasets like ogbn-arxiv."
        },
    )


@dataclass
class TrainingArguments(tmdlm.TMDLMConfig):
    output_dir: str = ".models/tmdlm"
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    # TM-DLM specific
    cls_loss_weight: float = 1.0
    ms_threshold: float = 0.0  # Set to 0.7 to enable TM-DLM-MS


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # --- Model & tokenizer ---
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # --- Dataset ---
    # Skip local_main_process_first() globally: every rank builds its own
    # dataset in parallel. Two reasons this is safe now:
    #   (1) LP-LLaGA path keeps Dataset.from_list(samples) in memory only.
    #   (2) NC path (load_tag_dataset) writes a TAG_CACHE_ROOT disk cache,
    #       but _tag_save_to_cache now gates the write to RANK==0 only so
    #       concurrent ranks never race on save_to_disk. Other ranks keep
    #       their in-memory copy.
    # Result: 8-rank-per-node parallel build instead of stage-1 / stage-2
    # serialization (~50% wall reduction on large datasets like arxiv).
    ds_arg = data_args.dataset_name
    if isinstance(ds_arg, str) and "," in ds_arg:
        ds_arg = [s.strip() for s in ds_arg.split(",") if s.strip()]

    if data_args.task == "lp":
        if not isinstance(ds_arg, str):
            raise ValueError(
                "LP task only supports a single dataset_name; got list "
                f"{ds_arg!r}"
            )
        # The auxiliary cls loss in TMDLMTrainer restricts logits to
        # digit tokens "0".."K-1", which does not match the LP answer
        # set (" yes"/" no"). Force-disable it; the main masked-token
        # CE on the answer position already supervises yes/no.
        if training_args.cls_loss_weight > 0:
            logger.warning(
                "Disabling cls_loss_weight for LP task (was %.2f); "
                "main MDLM loss on the answer token still supervises yes/no.",
                training_args.cls_loss_weight,
            )
            training_args.cls_loss_weight = 0.0
        _lp_kwargs = dict(
            tokenizer=tokenizer,
            max_seq_len=data_args.max_seq_len,
            max_neighbors_per_hop=data_args.max_neighbors_per_hop,
            max_hops=data_args.max_hops,
            mask_target_text=data_args.mask_target_text,
            neg_ratio=data_args.lp_neg_ratio,
            hard_neg_ratio=data_args.lp_hard_neg_ratio,
            use_llaga_split=data_args.lp_use_llaga_split,
        )
        train_dataset = load_lp_dataset(
            ds_arg, split="train",
            max_samples=data_args.max_train_samples,
            **_lp_kwargs,
        )
        # LLaGA's release ships only train + test JSONL (no val split).
        # When lp_use_llaga_split is set, the original code re-built the full
        # train split a second time as val_dataset — that doubles dataset
        # build time (≈75 min extra on arxiv) AND ignores max_train_samples,
        # so a 10%-data run still incurs the full val build. eval_lp_llaga_*.py
        # is the canonical held-out eval, so val_dataset is unused at
        # training time. Set it to None when LLaGA-split and eval is off.
        if data_args.lp_use_llaga_split:
            val_dataset = None
        else:
            val_dataset = load_lp_dataset(
                ds_arg, split="val", **_lp_kwargs,
            )
    else:
        _common_kwargs = dict(
            tokenizer=tokenizer,
            max_seq_len=data_args.max_seq_len,
            max_neighbors_per_hop=data_args.max_neighbors_per_hop,
            max_hops=data_args.max_hops,
            mask_target_text=data_args.mask_target_text,
            prompt_format=data_args.prompt_format,
            answer_label_style=data_args.answer_label_style,
            max_answer_tokens=data_args.max_answer_tokens,
            include_neighbor_labels=data_args.include_neighbor_labels,
            neighbor_label_format=data_args.neighbor_label_format,
            mask_neighbor_labels=data_args.mask_neighbor_labels,
            balance_merged=data_args.balance_merged,
            resample_strategy=data_args.resample_strategy,
            boost_spec=data_args.boost_spec,
        )
        train_dataset = load_tag_dataset(
            ds_arg, split="train",
            max_samples=data_args.max_train_samples,
            **_common_kwargs,
        )
        val_dataset = load_tag_dataset(
            ds_arg, split="val", **_common_kwargs
        )

    # --- Trainer ---
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training TM-DLM on %s...", data_args.dataset_name)
    trainer = tmdlm.TMDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=tmdlm.utils.GraphDataCollator(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
            position_id_type=data_args.position_id_type,
            use_topology_mask=data_args.use_topology_mask,
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
