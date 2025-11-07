"""Compare evaluation metrics between two adapters and write a summary report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence


DEFAULT_ADAPTER_A = Path("artifacts/results/adapter_v1/metrics.json")
DEFAULT_ADAPTER_B = Path("artifacts/results/adapter_v2/metrics.json")
DEFAULT_OUTPUT_JSON = Path("artifacts/results/adapter_comparison.json")
DEFAULT_OUTPUT_MD = Path("artifacts/results/adapter_comparison.md")

PRIMARY_KEYS = [
    "json_valid_pct",
    "schema_valid_pct",
    "answer_match_pct",
    "numeric_clean_pct",
    "avg_distractor_unique_ratio",
    "avg_prompt_tokens",
    "avg_completion_tokens",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare adapter evaluation metrics")
    parser.add_argument("--adapter-a", type=Path, default=DEFAULT_ADAPTER_A)
    parser.add_argument("--adapter-b", type=Path, default=DEFAULT_ADAPTER_B)
    parser.add_argument("--label-a", default="adapter_v1")
    parser.add_argument("--label-b", default="adapter_v2")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args(argv)


def load_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def extract_primary(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: float(metrics.get(key, 0.0)) for key in PRIMARY_KEYS}


def write_json(
    path: Path,
    label_a: str,
    label_b: str,
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    deltas: Dict[str, float],
) -> None:
    payload = {
        "adapter_a": label_a,
        "adapter_b": label_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "delta_b_minus_a": deltas,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(
    path: Path,
    label_a: str,
    label_b: str,
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    deltas: Dict[str, float],
) -> None:
    lines = [
        f"# Adapter Comparison: {label_a} vs {label_b}",
        "",
        "| Metric | "
        f"{label_a} | {label_b} | Δ ({label_b} - {label_a}) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in PRIMARY_KEYS:
        a = metrics_a.get(key, 0.0)
        b = metrics_b.get(key, 0.0)
        delta = deltas.get(key, 0.0)
        lines.append(f"| `{key}` | {a:.4f} | {b:.4f} | {delta:+.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metrics_a = load_metrics(args.adapter_a)
    metrics_b = load_metrics(args.adapter_b)
    primary_a = extract_primary(metrics_a)
    primary_b = extract_primary(metrics_b)
    deltas = {
        key: primary_b.get(key, 0.0) - primary_a.get(key, 0.0)
        for key in PRIMARY_KEYS
    }
    write_json(args.output_json, args.label_a, args.label_b, primary_a, primary_b, deltas)
    write_markdown(args.output_md, args.label_a, args.label_b, primary_a, primary_b, deltas)
    print(f"Wrote comparison to {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()


