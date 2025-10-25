"""Evaluate the graph summary LoRA adapter on the human-labelled set.

The script reloads the Qwen2.5-VL-7B-Instruct base model, attaches the LoRA
adapter produced by ``scripts/train_graph_model.py``, runs greedy/beam-free
generation over the held-out evaluation split, and computes lightweight
textual metrics (Rouge-L, unigram BLEU, cosine similarity).

Outputs:
    • ``results/graph_reader_v1/metrics.json`` – aggregate metrics
    • ``results/graph_reader_v1/samples.jsonl`` – per-sample metadata and scores

Example (PowerShell):

    & .\.venv312\Scripts\python.exe scripts\eval_graph_model.py \
        --graph-dir data/graph_human_descriptions/AQA \
        --adapter-dir models/adapters/graph_reader_v1 \
        --results-dir results/graph_reader_v1

"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from peft import PeftModel
from PIL import Image


SYSTEM_PROMPT = (
    "You are an AQA physics tutor. Describe each figure accurately in plain "
    "English without inventing features."
)


@dataclass
class EvalSample:
    image_path: Path
    prompt_text: str
    target_text: str
    metadata: Dict[str, object]


def discover_eval_samples(graph_dir: Path, eval_count: int) -> List[EvalSample]:
    json_paths = sorted(graph_dir.glob("*.json"))
    if len(json_paths) < eval_count:
        raise ValueError(
            f"Requested eval_count={eval_count} but only {len(json_paths)} samples available"
        )
    eval_paths = json_paths[-eval_count:]
    samples: List[EvalSample] = []
    for json_path in eval_paths:
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
        samples.append(
            EvalSample(
                image_path=image_path,
                prompt_text=prompt,
                target_text=target.strip(),
                metadata=metadata,
            )
        )
    return samples


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def tokenise(text: str) -> List[str]:
    return text.lower().replace("\n", " ").split()


def rouge_l(reference: List[str], candidate: List[str]) -> float:
    if not reference or not candidate:
        return 0.0

    m, n = len(reference), len(candidate)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if reference[i] == candidate[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    beta = 1.0
    score = ((1 + beta ** 2) * precision * recall) / (
        recall + beta ** 2 * precision
    )
    return score


def unigram_bleu(reference: List[str], candidate: List[str]) -> float:
    if not candidate:
        return 0.0
    ref_counts = Counter(reference)
    cand_counts = Counter(candidate)
    overlap = sum(min(count, ref_counts[token]) for token, count in cand_counts.items())
    precision = overlap / len(candidate)
    if precision == 0:
        return 0.0
    ref_len = len(reference)
    cand_len = len(candidate)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    return bp * precision


def cosine_similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> float:
    a = embedding_a / embedding_a.norm(dim=-1, keepdim=True)
    b = embedding_b / embedding_b.norm(dim=-1, keepdim=True)
    return float((a * b).sum())


def generate_summary(
    model: Qwen2_5_VLForConditionalGeneration,
    processor,
    sample: EvalSample,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": sample.prompt_text},
            ],
        },
    ]
    with Image.open(sample.image_path) as img:
        pil_image = img.convert("RGB")

    chat_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    processed_inputs = processor(
        text=[chat_prompt],
        images=[pil_image],
        return_tensors="pt",
        padding=False,
    )
    model_inputs = {}
    for key, value in processed_inputs.items():
        if torch.is_tensor(value):
            model_inputs[key] = value.to(device)

    input_length = model_inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
        )
    generated_text = processor.tokenizer.decode(
        generated[0][input_length:],
        skip_special_tokens=True,
    )
    return generated_text.strip()


def evaluate(args: argparse.Namespace) -> None:
    setup_logging(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    samples = discover_eval_samples(args.graph_dir, args.eval_count)
    logging.info("Loaded %d evaluation samples", len(samples))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    embedder = SentenceTransformer(args.embedding_model)

    predictions: List[Dict[str, object]] = []
    cos_scores: List[float] = []
    rouge_scores: List[float] = []
    bleu_scores: List[float] = []

    for sample in samples:
        prediction = generate_summary(
            model=model,
            processor=processor,
            sample=sample,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        ref_tokens = tokenise(sample.target_text)
        pred_tokens = tokenise(prediction)
        rouge = rouge_l(ref_tokens, pred_tokens)
        bleu = unigram_bleu(ref_tokens, pred_tokens)
        ref_embedding = embedder.encode(sample.target_text, convert_to_tensor=True)
        pred_embedding = embedder.encode(prediction, convert_to_tensor=True)
        cos = cosine_similarity(ref_embedding, pred_embedding)

        rouge_scores.append(rouge)
        bleu_scores.append(bleu)
        cos_scores.append(cos)

        predictions.append(
            {
                "image": str(sample.image_path.relative_to(args.graph_dir)),
                "prompt": sample.prompt_text,
                "target": sample.target_text,
                "prediction": prediction,
                "metrics": {
                    "rouge_l": rouge,
                    "bleu1": bleu,
                    "cosine": cos,
                    "cosine_below_threshold": cos < args.cosine_threshold,
                },
                "metadata": sample.metadata,
            }
        )

    metrics = {
        "mean_rouge_l": sum(rouge_scores) / len(rouge_scores),
        "mean_bleu1": sum(bleu_scores) / len(bleu_scores),
        "mean_cosine": sum(cos_scores) / len(cos_scores),
        "cosine_below_threshold": sum(c < args.cosine_threshold for c in cos_scores),
        "cosine_threshold": args.cosine_threshold,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.results_dir / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for entry in predictions:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logging.info("Evaluation complete. Metrics: %s", metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the graph LoRA adapter")
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graph_human_descriptions/AQA"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("models/adapters/graph_reader_v1"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/graph_reader_v1"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/eval/graph_reader_v1"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cosine-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":  # pragma: no cover
    main()

