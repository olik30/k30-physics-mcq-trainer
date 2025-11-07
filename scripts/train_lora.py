"""LoRA fine-tuning entrypoint for the Physics MCQ generator.

This script loads the chat-formatted dataset produced by ``scripts/format_for_sft.py``
and performs a supervised fine-tuning run using PEFT/LoRA on top of a base causal
language model (defaults to ``Qwen/Qwen2.5-7B-Instruct``).  Training progress is
streamed to a JSONL log and the resulting adapter weights are stored under
``models/adapters/<adapter_name>/``.  After every run, ``config/adapters.yaml`` is
updated so downstream tooling knows which adapter and hyperparameters were used.

The script is intentionally configurable so the same code path can run quick
sanity checks (tiny models, a single optimisation step) as well as long, fully
automated training jobs on the target 7B model.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "PyYAML is required for scripts/train_lora.py. Install it with 'pip install pyyaml'."
    ) from exc

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger(__name__)


DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class TrainingSummary:
    """Serializable summary for adapter cataloguing."""

    base_model: str
    adapter_name: str
    output_dir: str
    run_name: str
    train_file: str
    val_file: Optional[str]
    manifest_path: Optional[str]
    log_path: str
    hyperparameters: Dict[str, Any]
    train_metrics: Dict[str, Any]
    eval_metrics: Dict[str, Any]
    started_at: str
    finished_at: str


class JsonlTrainerLogger(TrainerCallback):
    """Write Trainer log events as structured JSON lines."""

    def __init__(self, log_path: Path, metadata: Dict[str, Any]):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.log_path.open("a", encoding="utf-8")
        self._write_event("trainer_init", metadata)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if not logs:
            return
        payload = {
            "step": state.global_step,
            "epoch": state.epoch,
            "metrics": logs,
        }
        self._write_event("trainer_log", payload)

    def on_train_end(self, args, state, control, **kwargs):  # noqa: D401
        self._write_event(
            "trainer_end",
            {
                "step": state.global_step,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
            },
        )
        self._stream.flush()
        self._stream.close()

    def _write_event(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        self._stream.write(json.dumps(record, ensure_ascii=False))
        self._stream.write("\n")
        self._stream.flush()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA adapters")
    parser.add_argument("--train-file", type=Path, default=Path("data/formatted/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("data/formatted/val.jsonl"))
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("models/adapters/adapter_v1"))
    parser.add_argument("--adapter-name", type=str, default="adapter_v1")
    parser.add_argument("--run-name", type=str, default="adapter_v1")
    parser.add_argument("--manifest", type=Path, default=Path("data/formatted/manifest.json"))
    parser.add_argument("--log-path", type=Path, default=Path("artifacts/logs/train/adapter_v1.jsonl"))

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0, help="Override epoch-based schedule")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        default=list(DEFAULT_LORA_TARGETS),
        help="Modules to wrap with LoRA (defaults match Qwen linear layers)",
    )

    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 training")
    parser.add_argument("--fp16", action="store_true", help="Use float16 training")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-eval", action="store_true", help="Skip validation pass")
    parser.add_argument("--pad-to-max-length", action="store_true", help="Pad sequences to max length")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true", help="Forward Trainer push_to_hub flag")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model/tokenizer code")
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device placement strategy passed to from_pretrained (e.g. 'auto', 'cuda', 'cpu')",
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("config/adapters.yaml"),
        help="Catalog file to update after training",
    )

    return parser.parse_args(argv)


def validate_paths(args: argparse.Namespace) -> None:
    if not args.train_file.exists():
        raise FileNotFoundError(f"--train-file does not exist: {args.train_file}")
    if args.val_file and not args.val_file.exists():
        LOGGER.warning("Validation file %s was not found; evaluation will be skipped.", args.val_file)
        args.no_eval = True
    if args.manifest and not args.manifest.exists():
        LOGGER.warning("Manifest file %s not found; continuing without dataset hash metadata.", args.manifest)


def load_conversation_dataset(train_file: Path, val_file: Optional[Path]) -> Dict[str, Any]:
    data_files = {"train": str(train_file)}
    if val_file and val_file.exists():
        data_files["validation"] = str(val_file)
    dataset = load_dataset("json", data_files=data_files)
    return dataset


def prepare_tokenizer(model_name: str, trust_remote_code: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(
    dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    pad_to_max_length: bool,
) -> Dict[str, Any]:
    remove_columns = None
    if "train" in dataset:
        remove_columns = dataset["train"].column_names
    elif "validation" in dataset:
        remove_columns = dataset["validation"].column_names

    def _tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Each record must contain a 'messages' list produced by format_for_sft.py")
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            chat_text,
            max_length=max_seq_length,
            truncation=True,
            padding=("max_length" if pad_to_max_length else False),
        )
        input_ids = tokenized["input_ids"]
        if pad_to_max_length:
            pad_id = tokenizer.pad_token_id
            labels = [token if token != pad_id else -100 for token in input_ids]
        else:
            labels = list(input_ids)
        tokenized["labels"] = labels
        return tokenized

    return dataset.map(_tokenize, remove_columns=remove_columns, desc="Tokenising conversations")


def init_model(args: argparse.Namespace) -> AutoModelForCausalLM:
    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    LOGGER.info("Loading base model %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        LOGGER.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


def wrap_peft_model(model: AutoModelForCausalLM, args: argparse.Namespace) -> AutoModelForCausalLM:
    LOGGER.info(
        "Attaching LoRA adapters (r=%s, alpha=%s, dropout=%.3f)",
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
    )
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target_modules,
    )
    lora_model = get_peft_model(model, lora_cfg)
    lora_model.print_trainable_parameters()
    return lora_model


def build_trainer(
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    trainer_logger: JsonlTrainerLogger,
) -> Trainer:
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    has_validation = "validation" in dataset and not args.no_eval
    evaluation_strategy = "no"
    if has_validation:
        evaluation_strategy = "steps"

    save_strategy = "steps"

    max_steps = args.max_steps if args.max_steps > 0 else -1

    use_cpu = str(args.device_map).lower().startswith("cpu")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy=evaluation_strategy,
        eval_steps=args.eval_steps if has_validation else None,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=max_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        no_cuda=use_cpu,
        use_cpu=use_cpu,
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        push_to_hub=args.push_to_hub,
        seed=args.seed,
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_validation else None,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[trainer_logger],
    )
    return trainer


def run_training(trainer: Trainer, args: argparse.Namespace) -> Dict[str, Any]:
    LOGGER.info("Starting training run")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model()
    trainer.save_state()
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    if trainer.args.push_to_hub:
        trainer.push_to_hub()

    eval_metrics: Dict[str, Any] = {}
    if trainer.eval_dataset is not None:
        LOGGER.info("Running evaluation on validation split")
        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics:
            eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    train_metrics = trainer.state.log_history[-1] if trainer.state.log_history else train_metrics
    return {"train": train_metrics, "eval": eval_metrics}


def summarise_run(
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
) -> TrainingSummary:
    hyperparameters = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
    }

    summary = TrainingSummary(
        base_model=args.base_model,
        adapter_name=args.adapter_name,
        output_dir=str(args.output_dir.resolve()),
        run_name=args.run_name,
        train_file=str(args.train_file.resolve()),
        val_file=str(args.val_file.resolve()) if args.val_file and args.val_file.exists() else None,
        manifest_path=str(args.manifest.resolve()) if args.manifest and args.manifest.exists() else None,
        log_path=str(args.log_path.resolve()),
        hyperparameters=hyperparameters,
        train_metrics=metrics.get("train", {}),
        eval_metrics=metrics.get("eval", {}),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
    )
    return summary


def update_adapter_catalog(config_path: Path, summary: TrainingSummary) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_data: Dict[str, Any]
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as stream:
            config_data = yaml.safe_load(stream) or {}
    else:
        config_data = {}

    adapters = config_data.setdefault("adapters", {})
    adapters[summary.adapter_name] = asdict(summary)

    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config_data, stream, sort_keys=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if args.device_map == "auto":
        if torch.cuda.is_available():
            args.device_map = "cuda"
        else:
            LOGGER.warning("CUDA not available; training will run on CPU.")
            args.device_map = "cpu"
    elif str(args.device_map).lower().startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU.")
        args.device_map = "cpu"

    validate_paths(args)

    set_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = load_conversation_dataset(args.train_file, None if args.no_eval else args.val_file)
    tokenizer = prepare_tokenizer(args.base_model, trust_remote_code=args.trust_remote_code)
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        pad_to_max_length=args.pad_to_max_length,
    )

    model = init_model(args)
    lora_model = wrap_peft_model(model, args)

    trainer_logger = JsonlTrainerLogger(
        log_path=args.log_path,
        metadata={
            "adapter": args.adapter_name,
            "base_model": args.base_model,
            "train_file": str(args.train_file),
            "val_file": str(args.val_file) if args.val_file else None,
            "run_name": args.run_name,
        },
    )

    trainer = build_trainer(args, lora_model, tokenizer, tokenized_dataset, trainer_logger)

    start_time = datetime.now(timezone.utc)
    metrics = run_training(trainer, args)
    finish_time = datetime.now(timezone.utc)

    summary = summarise_run(args, metrics, start_time, finish_time)
    update_adapter_catalog(args.config_path, summary)

    LOGGER.info("Training complete. Adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("Training interrupted by user")
        sys.exit(1)

