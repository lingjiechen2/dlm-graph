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

import accelerate
import transformers

import dllm
from dllm.pipelines import tmdlm
from dllm.data.graph import load_tag_dataset

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="cora",
        metadata={"help": "TAG dataset: cora | pubmed | ogbn-arxiv | ogbn-products"},
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
    with accelerate.PartialState().local_main_process_first():
        train_dataset = load_tag_dataset(
            data_args.dataset_name,
            tokenizer=tokenizer,
            split="train",
            max_seq_len=data_args.max_seq_len,
            max_neighbors_per_hop=data_args.max_neighbors_per_hop,
            max_hops=data_args.max_hops,
            mask_target_text=data_args.mask_target_text,
        )
        val_dataset = load_tag_dataset(
            data_args.dataset_name,
            tokenizer=tokenizer,
            split="val",
            max_seq_len=data_args.max_seq_len,
            max_neighbors_per_hop=data_args.max_neighbors_per_hop,
            max_hops=data_args.max_hops,
            mask_target_text=data_args.mask_target_text,
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
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
