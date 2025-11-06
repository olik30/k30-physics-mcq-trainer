"""Evaluate LoRA-adapted MCQ model outputs against the test split.

This script loads the chat-formatted dataset produced by ``scripts/format_for_sft.py``
and, for each entry, prompts the specified base model plus LoRA adapter to generate
an answer.  It then parses the JSON output, checks schema validity using the existing
Day 9 validator, and computes a collection of heuristics:

* JSON validity / schema compliance
* Option diversity and duplicate detection
* Answer agreement with the reference item
* Presence of suspicious scientific-notation placeholders
* AO distribution alignment versus the reference set
* Hint/explanation length heuristics

Results are written to ``results/<adapter_name>/metrics.json`` with a line-per-sample
dump stored in ``results/<adapter_name>/samples.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:  # pragma: no cover
    from . import filter_balance, auto_generate
except ImportError:  # pragma: no cover
    import filter_balance  # type: ignore
    import auto_generate  # type: ignore


REQUIRED_KEYS = {
    "question",
    "options",
    "correct_index",
    "hint",
    "explanation",
    "ao",
    "topic",
    "difficulty",
}


@dataclass
class SampleResult:
    """Holds evaluation artefacts for a single record."""

    record_id: str
    prompt_tokens: int
    completion_tokens: int
    raw_output: str
    parsed: Optional[Dict[str, Any]]
    parse_error: Optional[str]
    schema_errors: List[str]
    reference: Dict[str, Any]
    json_valid: bool
    schema_valid: bool
    answer_match: bool
    distractor_unique_ratio: float
    numeric_ok: bool
    hint_chars: int
    explanation_chars: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA-adapted MCQ model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/formatted/test.jsonl"),
        help="Chat-formatted dataset to evaluate",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base causal LM to load before applying the adapter",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("models/adapters/adapter_v1"),
        help="Directory containing the trained LoRA adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adapter_v1"),
        help="Directory to store evaluation artefacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model inference (e.g. 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate for each completion",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model/tokenizer code",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of samples to evaluate",
    )
    return parser.parse_args(argv)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return records


def parse_reference(record: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the ground-truth assistant message."""
    messages = record.get("messages") or []
    if len(messages) < 2:
        raise ValueError(f"Record {record.get('id')} missing assistant message")
    assistant = messages[-1]
    content = assistant.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Assistant content is not string (id={record.get('id')})")
    return json.loads(content)


def extract_json_payload(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Attempt to parse JSON payload from the generated text."""
    text = text.strip()
    if not text:
        return None, "empty_output"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet), None
            except json.JSONDecodeError as exc:
                return None, f"json_decode_error: {exc.msg}"
        return None, "json_braces_not_found"


def prepare_tokenizer(model_name: str, trust_remote_code: bool) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def prepare_model(
    model_name: str,
    adapter_path: Path,
    device: str,
    trust_remote_code: bool,
) -> AutoModelForCausalLM:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    peft_model.eval()
    return peft_model


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_messages: List[Dict[str, str]],
    generation_config: GenerationConfig,
    device: str,
) -> Tuple[str, int, int]:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    generated_tokens = output[0][inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    prompt_tokens = inputs["input_ids"].shape[1]
    completion_tokens = generated_tokens.shape[0]
    return completion, prompt_tokens, completion_tokens


def evaluate_sample(
    record: Dict[str, Any],
    reference: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
) -> Tuple[bool, bool, bool, float, bool, int, int, List[str]]:
    schema_errors: List[str] = []
    json_valid = prediction is not None
    schema_valid = False
    answer_match = False
    unique_ratio = 0.0
    numeric_ok = True
    hint_chars = 0
    explanation_chars = 0

    if prediction is not None:
        missing = REQUIRED_KEYS - set(prediction)
        if missing:
            schema_errors.append(f"missing_keys:{','.join(sorted(missing))}")
        else:
            processed = filter_balance.analyse_record(prediction)
            schema_errors.extend(processed.errors)
            schema_errors.extend(processed.warnings)
            schema_valid = not processed.errors

            options = prediction.get("options") or []
            if isinstance(options, list) and options:
                unique_ratio = len(set(opt.strip() for opt in options if isinstance(opt, str))) / max(len(options), 1)
            else:
                unique_ratio = 0.0

            correct_index = prediction.get("correct_index")
            if (
                isinstance(options, list)
                and isinstance(reference.get("options"), list)
                and isinstance(correct_index, int)
                and 0 <= correct_index < len(options)
                and len(reference["options"]) == len(options)
            ):
                pred_answer = options[correct_index].strip()
                ref_answer = reference["options"][reference["correct_index"]].strip()
                answer_match = pred_answer == ref_answer

            numeric_ok = not _contains_numeric_placeholder(prediction)
            hint_chars = len(str(prediction.get("hint") or "")) if prediction.get("hint") is not None else 0
            explanation_chars = (
                len(str(prediction.get("explanation") or ""))
                if prediction.get("explanation") is not None
                else 0
            )

    return (
        json_valid,
        schema_valid,
        answer_match,
        unique_ratio,
        numeric_ok,
        hint_chars,
        explanation_chars,
        schema_errors,
    )


def _contains_numeric_placeholder(prediction: Dict[str, Any]) -> bool:
    fields = []
    for key in ("question", "hint", "explanation"):
        value = prediction.get(key)
        if isinstance(value, str):
            fields.append(value)
    options = prediction.get("options")
    if isinstance(options, list):
        fields.extend(str(opt) for opt in options if isinstance(opt, str))
    for field in fields:
        if auto_generate.BROKEN_SCI_NOTATION_RE.search(field) or auto_generate.PLACEHOLDER_RE.search(field):
            return True
    return False


def aggregate_metrics(samples: Iterable[SampleResult], ao_reference: Counter) -> Dict[str, Any]:
    total = 0
    json_valid = 0
    schema_valid = 0
    answer_match = 0
    numeric_clean = 0
    unique_ratios: List[float] = []
    hint_lengths: List[int] = []
    explanation_lengths: List[int] = []
    completion_tokens = 0
    prompt_tokens = 0
    ao_counter = Counter()
    hint_short = 0

    for sample in samples:
        total += 1
        prompt_tokens += sample.prompt_tokens
        completion_tokens += sample.completion_tokens
        if sample.json_valid:
            json_valid += 1
        if sample.schema_valid:
            schema_valid += 1
        if sample.answer_match:
            answer_match += 1
        if sample.numeric_ok:
            numeric_clean += 1
        unique_ratios.append(sample.distractor_unique_ratio)
        hint_lengths.append(sample.hint_chars)
        explanation_lengths.append(sample.explanation_chars)
        ao_tag = (sample.parsed or {}).get("ao")
        if isinstance(ao_tag, str):
            ao_counter[ao_tag] += 1
        if sample.hint_chars < 40:
            hint_short += 1

    def ratio(count: int) -> float:
        return count / total if total else 0.0

    ao_alignment = {}
    all_aos = set(ao_counter) | set(ao_reference)
    for ao in all_aos:
        pct_model = ao_counter[ao] / total if total else 0.0
        pct_ref = ao_reference[ao] / sum(ao_reference.values()) if ao_reference else 0.0
        ao_alignment[ao] = {"model_pct": pct_model, "reference_pct": pct_ref, "abs_diff": abs(pct_model - pct_ref)}

    avg_unique = statistics.fmean(unique_ratios) if unique_ratios else 0.0
    avg_hint_chars = statistics.fmean(hint_lengths) if hint_lengths else 0.0
    median_hint_chars = statistics.median(hint_lengths) if hint_lengths else 0.0
    avg_expl_chars = statistics.fmean(explanation_lengths) if explanation_lengths else 0.0

    metrics: Dict[str, Any] = {
        "total_records": total,
        "json_valid_pct": ratio(json_valid),
        "schema_valid_pct": ratio(schema_valid),
        "answer_match_pct": ratio(answer_match),
        "numeric_clean_pct": ratio(numeric_clean),
        "avg_distractor_unique_ratio": avg_unique,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "avg_prompt_tokens": prompt_tokens / total if total else 0.0,
        "avg_completion_tokens": completion_tokens / total if total else 0.0,
        "ao_distribution": ao_counter,
        "ao_alignment": ao_alignment,
        "hint_quality": {
            "avg_chars": avg_hint_chars,
            "median_chars": median_hint_chars,
            "short_hint_pct": ratio(hint_short),
        },
        "explanation_avg_chars": avg_expl_chars,
    }
    return metrics


def _serialise_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert non-JSON-native objects to plain Python types."""
    serialised: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, Counter):
            serialised[key] = dict(value)
        elif isinstance(value, dict):
            serialised[key] = _serialise_metrics(value)
        else:
            serialised[key] = value
    return serialised


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    records = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    device = args.device
    tokenizer = prepare_tokenizer(args.base_model, args.trust_remote_code)
    model = prepare_model(args.base_model, args.adapter_path, device, args.trust_remote_code)

    do_sample = args.temperature > 0
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p if do_sample else None,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    samples: List[SampleResult] = []
    ao_reference_counter: Counter = Counter()

    for record in tqdm(records, desc="Evaluating"):
        record_id = record.get("id") or "unknown"
        reference = parse_reference(record)
        ao_reference_counter[reference.get("ao")] += 1

        messages = record.get("messages") or []
        prompt_messages = messages[:-1]

        generated_text, prompt_tokens, completion_tokens = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt_messages=prompt_messages,
            generation_config=generation_config,
            device=device,
        )

        parsed, parse_error = extract_json_payload(generated_text)
        (
            json_valid,
            schema_valid,
            answer_match,
            unique_ratio,
            numeric_ok,
            hint_chars,
            explanation_chars,
            schema_errors,
        ) = evaluate_sample(record, reference, parsed)

        sample = SampleResult(
            record_id=record_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_output=generated_text,
            parsed=parsed,
            parse_error=parse_error,
            schema_errors=schema_errors,
            reference=reference,
            json_valid=json_valid,
            schema_valid=schema_valid,
            answer_match=answer_match,
            distractor_unique_ratio=unique_ratio,
            numeric_ok=numeric_ok,
            hint_chars=hint_chars,
            explanation_chars=explanation_chars,
        )
        samples.append(sample)

    metrics = aggregate_metrics(samples, ao_reference_counter)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    metrics_serialisable = _serialise_metrics(metrics)
    metrics_path.write_text(json.dumps(metrics_serialisable, indent=2), encoding="utf-8")

    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as stream:
        for sample in samples:
            data = {
                "id": sample.record_id,
                "prompt_tokens": sample.prompt_tokens,
                "completion_tokens": sample.completion_tokens,
                "raw_output": sample.raw_output,
                "parse_error": sample.parse_error,
                "schema_errors": sample.schema_errors,
                "json_valid": sample.json_valid,
                "schema_valid": sample.schema_valid,
                "answer_match": sample.answer_match,
                "distractor_unique_ratio": sample.distractor_unique_ratio,
                "numeric_ok": sample.numeric_ok,
                "hint_chars": sample.hint_chars,
                "explanation_chars": sample.explanation_chars,
            }
            if sample.parsed is not None:
                data["parsed"] = sample.parsed
            stream.write(json.dumps(data, ensure_ascii=False))
            stream.write("\n")

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote sample outputs to {samples_path}")


if __name__ == "__main__":
    main()

