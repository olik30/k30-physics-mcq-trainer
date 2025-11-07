"""Create a fixed evaluation deck for the MCQ reviewer retrain loop.

This script samples a balanced subset of already-approved MCQs, converts them to
chat format, and writes the result to ``data/eval/eval_set.jsonl`` together with
summary metadata.  The goal is to provide a stable, larger evaluation set so
each retrain cycle can report reliable metrics while humans focus on reviewing
new variants.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - script/package dual usage
    from . import filter_balance, format_for_sft
except ImportError:  # pragma: no cover
    import filter_balance  # type: ignore
    import format_for_sft  # type: ignore


DEFAULT_INPUTS = [
    Path("data/filtered/seed_train.filtered.jsonl"),
    Path("data/filtered/refresh_accept.jsonl"),
]

DEFAULT_MULTI_OUTPUT_DIR = Path("data/eval/core_sets")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stable evaluation dataset")
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Filtered MCQ JSONL files to sample from (defaults to seed + accepted variants)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=250,
        help="Target number of evaluation items (0 = use every valid record)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/eval_set.jsonl"),
        help="Destination JSONL for chat-formatted evaluation records",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/eval/manifest.json"),
        help="Path to write manifest / summary JSON",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/eval/stats.json"),
        help="Optional path to write distribution stats",
    )
    parser.add_argument(
        "--sets",
        type=int,
        default=1,
        help="Number of independent evaluation subsets to create (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for multi-set output (defaults to data/eval/core_sets when --sets > 1)",
    )
    parser.add_argument(
        "--label-prefix",
        type=str,
        default="core",
        help="Filename prefix for multi-set outputs (e.g., core_01.jsonl)",
    )
    return parser.parse_args(argv)


def _load_inputs(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"Skipping missing input: {path}")
            continue
        print(f"Loading records from {path}")
        records.extend(filter_balance.load_jsonl(path))
    if not records:
        raise SystemExit("No records found in the provided inputs.")
    return records


def _sanitise_records(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Counter]:
    clean: List[Dict[str, Any]] = []
    skipped = Counter()
    seen_ids = set()

    for raw in records:
        record_id = raw.get("id")
        if record_id in seen_ids:
            skipped["duplicate_id"] += 1
            continue

        processed = filter_balance.analyse_record(raw)
        if processed.errors:
            skipped["failed_validation"] += 1
            continue

        clean.append(processed.cleaned)
        seen_ids.add(record_id)

    if not clean:
        raise SystemExit("All records were filtered out; evaluation deck would be empty.")

    return clean, skipped


def _bucket_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    topic = str(record.get("topic") or "(missing)").strip().lower()
    difficulty = str(record.get("difficulty") or "(missing)").strip().lower()
    ao = str(record.get("ao") or "(missing)").strip().lower()
    return topic, difficulty, ao


def _stratified_sample(records: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    if limit <= 0 or limit >= len(records):
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        return shuffled

    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[_bucket_key(record)].append(record)

    for bucket_records in buckets.values():
        rng.shuffle(bucket_records)

    selected: List[Dict[str, Any]] = []
    bucket_order = list(buckets.keys())
    rng.shuffle(bucket_order)

    while bucket_order and len(selected) < limit:
        for key in bucket_order[:]:
            bucket = buckets[key]
            if not bucket:
                bucket_order.remove(key)
                continue
            selected.append(bucket.pop())
            if not bucket:
                bucket_order.remove(key)
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        remaining: List[Dict[str, Any]] = []
        for bucket in buckets.values():
            remaining.extend(bucket)
        rng.shuffle(remaining)
        for record in remaining:
            if len(selected) >= limit:
                break
            selected.append(record)

    return selected


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    raw_records = _load_inputs(args.inputs)
    clean_records, skipped = _sanitise_records(raw_records)

    print(
        f"Sanitised {len(clean_records)} records (skipped {skipped['failed_validation']} validation failures,"
        f" {skipped['duplicate_id']} duplicates)."
    )

    if args.sets <= 1:
        selection = _stratified_sample(clean_records, args.count if args.count else len(clean_records), args.seed)
        print(f"Selected {len(selection)} records for evaluation deck (target {args.count}).")

        conversations = format_for_sft.convert_to_conversations(selection)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sha256 = format_for_sft.write_jsonl(args.output, conversations)
        print(f"Wrote evaluation set to {args.output} (sha256 {sha256})")

        stats = format_for_sft.compute_stats("eval", conversations)
        stats.update({"seed": args.seed, "source_files": [str(p) for p in args.inputs]})
        if args.stats_path:
            args.stats_path.parent.mkdir(parents=True, exist_ok=True)
            args.stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        manifest = {
            "output": str(args.output),
            "sha256": sha256,
            "count": len(conversations),
            "seed": args.seed,
            "source_files": [str(p) for p in args.inputs],
            "requested_count": args.count,
            "skipped": dict(skipped),
            "stats_path": str(args.stats_path),
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest written to {args.manifest}")
        return

    output_dir = args.output_dir or DEFAULT_MULTI_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.sets} evaluation sets in {output_dir}")

    aggregate_manifest: Dict[str, Any] = {
        "sets": [],
        "seed_base": args.seed,
        "requested_count": args.count,
        "source_files": [str(p) for p in args.inputs],
        "skipped": dict(skipped),
    }

    for idx in range(args.sets):
        subset_seed = args.seed + idx
        target_count = args.count if args.count else len(clean_records)
        if target_count > len(clean_records):
            target_count = len(clean_records)
        subset = _stratified_sample(clean_records, target_count, subset_seed)
        label = f"{args.label_prefix}_{idx + 1:02d}"
        output_path = output_dir / f"{label}.jsonl"
        stats_path = output_dir / f"{label}_stats.json"

        print(f"Set {idx + 1}/{args.sets}: selected {len(subset)} records (target {target_count}).")
        conversations = format_for_sft.convert_to_conversations(subset)
        sha256 = format_for_sft.write_jsonl(output_path, conversations)
        print(f"  Wrote evaluation set to {output_path} (sha256 {sha256})")

        stats = format_for_sft.compute_stats(label, conversations)
        stats.update({"seed": subset_seed, "source_files": [str(p) for p in args.inputs]})
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        entry = {
            "label": label,
            "path": str(output_path),
            "sha256": sha256,
            "count": len(conversations),
            "seed": subset_seed,
            "stats_path": str(stats_path),
        }
        aggregate_manifest["sets"].append(entry)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(aggregate_manifest, indent=2), encoding="utf-8")
    print(f"Multi-set manifest written to {manifest_path}")


if __name__ == "__main__":
    main()


