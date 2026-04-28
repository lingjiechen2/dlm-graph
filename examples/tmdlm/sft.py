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
    prompt_format: str = field(
        default="mc_digit",
        metadata={
            "help": "Prompt format: 'mc_digit' (digit answer) or 'category_infill' (class-name infill with eos padding)"
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
    use_topology_mask: bool = field(
        default=True,
        metadata={
            "help": "Apply star-topology attention mask restricting neighbor-target attention"
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
        )
        train_dataset = load_tag_dataset(
            data_args.dataset_name, split="train", **_common_kwargs
        )
        val_dataset = load_tag_dataset(
            data_args.dataset_name, split="val", **_common_kwargs
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
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
