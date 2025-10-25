"""LoRA fine-tuning entry point for the graph understanding mini-model.

This script adapts Qwen2.5-VL-7B-Instruct using the human-labelled graph
descriptions under ``data/graph_human_descriptions/AQA``.

Workflow (default settings):

1. Split the 40 vision-language tuples into a 32/8 train/eval partition.
2. Load the base VLM in 4-bit (QLoRA) with gradient checkpointing.
3. Optimise LoRA adapters over the human descriptions while keeping the base
   weights frozen.
4. Save the adapter weights to ``models/adapters/graph_reader_v1/`` and emit a
   structured training log under ``logs/train/graph_reader_v1/``.

All paths, hyperparameters, and split behaviour are configurable via CLI flags.

Example usage (PowerShell):

    & .\.venv312\Scripts\python.exe scripts\train_graph_model.py ^
        --graph-dir data/graph_human_descriptions/AQA ^
        --output-dir models/adapters/graph_reader_v1 ^
        --log-dir logs/train/graph_reader_v1 ^
        --epochs 3 --learning-rate 2e-4

The script never copies the PNG assets; it streams them from disk to minimise
VRAM and storage overhead.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence
import inspect

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    Qwen2_5_VLForConditionalGeneration,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from PIL import Image

# ---------------------------------------------------------------------------
# Data structures


SYSTEM_PROMPT = (
    "You are an AQA physics tutor. Describe each figure accurately in plain "
    "English without inventing features."
)


@dataclass
class GraphRecord:
    """Simple container for one human-labelled graph sample."""

    image_path: Path
    prompt_text: str
    target_text: str
    metadata: Dict[str, object]


class GraphHumanDataset(Dataset):
    """Map-style dataset that keeps only lightweight metadata in memory."""

    def __init__(
        self,
        records: Sequence[GraphRecord],
        processor: "AutoProcessor",
        tokenizer,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = self.records[index]
        with Image.open(rec.image_path) as img:
            pil_image = img.convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": rec.prompt_text or ""},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rec.target_text or ""}],
            },
        ]

        full_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt_messages = messages[:-1]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        processed_full = self.processor(
            text=[full_text], images=[pil_image], return_tensors="pt", padding=False
        )
        processed_prompt = self.processor(
            text=[prompt_text], images=[pil_image], return_tensors="pt", padding=False
        )

        input_ids = processed_full["input_ids"].squeeze(0)
        attention_mask = processed_full["attention_mask"].squeeze(0)
        pixel_values = processed_full["pixel_values"].squeeze(0)
        pixel_attention_mask = processed_full.get("pixel_attention_mask")
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.squeeze(0)
        image_grid_thw = processed_full.get("image_grid_thw")
        if image_grid_thw is not None:
            if image_grid_thw.ndim == 2 and image_grid_thw.shape[0] == 1:
                image_grid_thw = image_grid_thw.squeeze(0)
            image_grid_thw = image_grid_thw.to(dtype=torch.long)

        labels = input_ids.clone()

        prompt_length = processed_prompt["input_ids"].shape[-1]
        labels[:prompt_length] = -100
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels = labels.masked_fill(labels == pad_token_id, -100)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }

        if pixel_attention_mask is not None:
            batch["pixel_attention_mask"] = pixel_attention_mask
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw

        return batch


class VisionLanguageCollator:
    """Pad variable-length sequences while stacking image tensors."""

    def __init__(self, processor: "AutoProcessor", tokenizer) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id or 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        pixel_attention_mask = None
        if "pixel_attention_mask" in features[0]:
            pixel_attention_mask = torch.stack(
                [f["pixel_attention_mask"] for f in features]
            )
        image_grid_thw = None
        if "image_grid_thw" in features[0]:
            image_grid_thw = torch.stack(
                [f["image_grid_thw"].to(torch.long) for f in features]
            )

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        padding = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        labels_padding = self.tokenizer.pad(
            {"input_ids": labels}, padding=True, return_tensors="pt"
        )["input_ids"]
        labels_padding = labels_padding.masked_fill(
            labels_padding == self.pad_token_id, -100
        )

        batch = {
            "input_ids": padding["input_ids"],
            "attention_mask": padding["attention_mask"],
            "pixel_values": pixel_values,
            "labels": labels_padding,
        }

        if pixel_attention_mask is not None:
            batch["pixel_attention_mask"] = pixel_attention_mask
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw

        return batch


# ---------------------------------------------------------------------------
# Utilities


def discover_graph_records(graph_dir: Path, eval_count: int) -> Dict[str, List[GraphRecord]]:
    json_paths = sorted(graph_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {graph_dir}")

    if len(json_paths) < eval_count:
        raise ValueError(
            f"Requested eval_count={eval_count} but only {len(json_paths)} samples available"
        )

    train_records: List[GraphRecord] = []
    eval_records: List[GraphRecord] = []

    split_index = len(json_paths) - eval_count

    for idx, json_path in enumerate(json_paths):
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        image_name = payload.get("image")
        if not image_name:
            raise ValueError(f"Missing 'image' key in {json_path}")

        image_path = graph_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image referenced by {json_path} not found: {image_path}"
            )

        title = payload.get("title")
        prompt = (
            f"Figure title: {title}. Describe the visual trends clearly."
            if title
            else "Describe the key features of this diagram for AQA physics revision."
        )
        target = payload.get("human_description")
        if not target:
            raise ValueError(f"Missing 'human_description' in {json_path}")

        metadata = {
            "source_json": str(json_path.relative_to(graph_dir)),
            "title": title,
            "visual_content": payload.get("visual_content"),
            "gpt_description": payload.get("gpt_description"),
        }

        record = GraphRecord(
            image_path=image_path,
            prompt_text=prompt,
            target_text=target.strip(),
            metadata=metadata,
        )

        if idx < split_index:
            train_records.append(record)
        else:
            eval_records.append(record)

    return {"train": train_records, "eval": eval_records}


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.info("Logging to %s", log_path)
    return log_path


def find_linear_target_modules(model: torch.nn.Module) -> List[str]:
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            key = name.split(".")[-1]
            if any(token in key for token in ("q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj")):
                target_modules.add(key)
    if not target_modules:
        # Fallback to default QLoRA set for transformer blocks
        target_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    return sorted(target_modules)


# ---------------------------------------------------------------------------
# Training orchestration


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = discover_graph_records(args.graph_dir, args.eval_count)
    logging.info(
        "Loaded %d training and %d evaluation samples",
        len(records["train"]),
        len(records["eval"]),
    )

    dataset_train = GraphHumanDataset(records["train"], processor, tokenizer)
    dataset_eval = GraphHumanDataset(records["eval"], processor, tokenizer)

    collator = VisionLanguageCollator(processor, tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    target_modules = find_linear_target_modules(model)
    logging.info("Applying LoRA to modules: %s", ", ".join(target_modules))

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_train_batch = args.batch_size * args.gradient_accumulation_steps
    logging.info(
        "Effective batch size: %d (micro=%d, grad_accum=%d)",
        total_train_batch,
        args.batch_size,
        args.gradient_accumulation_steps,
    )

    warmup_steps = args.warmup_steps
    if warmup_steps < 0:
        warmup_steps = max(1, math.ceil(0.1 * len(dataset_train)))

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=1,
        logging_steps=1,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to=[],
        max_grad_norm=1.0,
    )

    training_sig = inspect.signature(TrainingArguments.__init__)
    params = training_sig.parameters
    if "evaluation_strategy" in params:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:
        training_kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in params:
        training_kwargs["save_strategy"] = "epoch"
    elif "save_strategy" not in params and "save_steps" in params:
        training_kwargs["save_steps"] = 0
    if "logging_steps" not in params and "log_steps" in params:
        training_kwargs["log_steps"] = 1

    training_args = TrainingArguments(**training_kwargs)

    run_meta = {
        "model_name": args.model_name,
        "train_samples": len(records["train"]),
        "eval_samples": len(records["eval"]),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        },
    }

    class JsonLogger(TrainerCallback):
        def __init__(self, run_log_path: Path) -> None:
            self.run_log_path = run_log_path
            self.entries: List[Dict[str, object]] = []

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs=None,
            **kwargs,
        ) -> None:
            if logs is None:
                return
            entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "logs": logs,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.entries.append(entry)

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> None:
            with self.run_log_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "meta": run_meta,
                        "metrics": self.entries,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=collator,
        callbacks=[JsonLogger(run_log_path)],
    )

    trainer.train()

    trainer.save_state()
    trainer.save_model()

    logging.info("Training finished. Adapter saved to %s", args.output_dir)


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL on graph summaries")
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graph_human_descriptions/AQA"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/adapters/graph_reader_v1"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/train/graph_reader_v1"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=-1, help="Negative -> auto 10%% of dataset")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-count", type=int, default=8, help="Number of samples reserved for evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(args.log_dir))
    run_training(args)


if __name__ == "__main__":  # pragma: no cover
    main()

