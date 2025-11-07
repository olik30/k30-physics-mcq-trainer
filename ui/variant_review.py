"""Streamlit UI for reviewing MCQ variants side-by-side."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st


DEFAULT_DATASET = Path("data/filtered/refresh_candidates.jsonl")
DEFAULT_LOG = Path("data/review/variant_choices.jsonl")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def determine_variant_group(record: Dict[str, Any]) -> Optional[str]:
    metadata = record.get("metadata") or {}
    group_id = metadata.get("variant_group_id")
    if group_id:
        return str(group_id)
    question_id = ""
    part_code = ""
    for source in record.get("sources") or []:
        if isinstance(source, dict) and source.get("type") == "question_part":
            question_id = str(source.get("question_id") or "")
            part_code = str(source.get("part_code") or "")
            break
    if question_id:
        return f"{question_id}#{part_code or ''}"
    return None


def resolve_variant_id(record: Dict[str, Any], fallback: int) -> int:
    metadata = record.get("metadata") or {}
    variant_id = record.get("variant_id") or metadata.get("variant_id")
    try:
        return int(variant_id)
    except (TypeError, ValueError):
        return fallback


def group_variants(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for record in records:
        group_id = determine_variant_group(record)
        if not group_id:
            continue
        question_id = None
        part_code = None
        for source in record.get("sources") or []:
            if isinstance(source, dict) and source.get("type") == "question_part":
                question_id = source.get("question_id")
                part_code = source.get("part_code")
                break
        group = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "question_id": question_id,
                "part_code": part_code,
                "records": [],
            },
        )
        variant_id = resolve_variant_id(record, len(group["records"]) + 1)
        group["records"].append({"variant_id": variant_id, "record": record})
        if group_id not in order:
            order.append(group_id)

    grouped_list: List[Dict[str, Any]] = []
    for group_id in order:
        group = grouped[group_id]
        group["records"] = sorted(group["records"], key=lambda item: item["variant_id"])
        grouped_list.append(group)
    return grouped_list


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_log_entry(log_path: Path, entry: Dict[str, Any]) -> None:
    ensure_parent(log_path)
    with log_path.open("a", encoding="utf-8") as stream:
        json.dump(entry, stream, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    st.set_page_config(page_title="MCQ Variant Reviewer", layout="wide")
    st.title("MCQ Variant Reviewer")

    default_dataset = str(DEFAULT_DATASET.resolve())
    default_log = str(DEFAULT_LOG.resolve())

    dataset_path = Path(st.sidebar.text_input("Dataset path", default_dataset))
    log_path = Path(st.sidebar.text_input("Log path", default_log))
    show_reviewed = st.sidebar.checkbox("Show already reviewed groups", value=False)

    records = load_jsonl(dataset_path)
    groups = group_variants(records)

    log_entries_raw = load_jsonl(log_path)
    reviewed_ids = {
        entry.get("variant_group_id")
        or (f"{entry.get('question_id')}#{entry.get('part_code') or ''}" if entry.get("question_id") else None)
        for entry in log_entries_raw
    }
    reviewed_ids = {gid for gid in reviewed_ids if gid}

    pending_groups = [group for group in groups if show_reviewed or group["group_id"] not in reviewed_ids]

    if "group_index" not in st.session_state:
        st.session_state.group_index = 0

    if not pending_groups:
        st.success("No variant groups left to review. 🎉")
        if not show_reviewed and reviewed_ids:
            st.info("Toggle 'Show already reviewed groups' to revisit completed items.")
        return

    index = st.session_state.group_index % len(pending_groups)
    group = pending_groups[index]
    group_id = group["group_id"]

    st.subheader(f"Group {index + 1} of {len(pending_groups)} — {group_id}")
    question_id = group.get("question_id")
    part_code = group.get("part_code")
    if question_id:
        st.write(f"**Question ID:** `{question_id}`  |  **Part:** `{part_code}`")

    cols = st.columns(len(group["records"]))
    for col, item in zip(cols, group["records"]):
        record = item["record"]
        variant_id = item["variant_id"]
        with col:
            st.markdown(f"### Variant {variant_id}")
            st.markdown(f"**Question**: {record.get('question')}")
            options = record.get("options") or []
            for idx, opt in enumerate(options):
                label = chr(ord("A") + idx)
                st.markdown(f"- **{label}.** {opt}")
            st.markdown(f"**Hint:** {record.get('hint')}")
            st.markdown(f"**Explanation:** {record.get('explanation')}")
            st.markdown(
                f"**Meta:** AO={record.get('ao')} | Topic={record.get('topic')} | Difficulty={record.get('difficulty')}"
            )

    with st.form(key=f"review_form_{group_id}"):
        preferred = st.radio(
            "Preferred variant",
            options=["None"] + [str(item["variant_id"]) for item in group["records"]],
            index=0,
            horizontal=True,
        )

        decisions: List[Dict[str, Any]] = []
        for item in group["records"]:
            variant_id = item["variant_id"]
            st.markdown(f"#### Variant {variant_id} decision")
            decision = st.radio(
                f"Decision for variant {variant_id}",
                options=["Accept", "Reject"],
                key=f"decision_{group_id}_{variant_id}",
            )
            note = st.text_area(
                f"Notes for variant {variant_id}",
                key=f"note_{group_id}_{variant_id}",
                placeholder="Why accepted/rejected? Why not preferred?",
            )
            decisions.append({
                "variant_id": variant_id,
                "decision": decision.lower(),
                "note": note.strip(),
            })

        submitted = st.form_submit_button("Save decisions")

    if submitted:
        preferred_variant = None if preferred == "None" else int(preferred)
        entry = {
            "variant_group_id": group_id,
            "question_id": question_id,
            "part_code": part_code,
            "preferred_variant": preferred_variant,
            "variants": [
                {
                    **variant,
                    "is_preferred": bool(preferred_variant and variant["variant_id"] == preferred_variant),
                }
                for variant in decisions
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
        append_log_entry(log_path, entry)
        st.success("Decisions saved.")
        st.session_state.group_index += 1
        st.experimental_rerun()

    col_prev, col_skip, col_next = st.columns([1, 1, 1])
    with col_prev:
        if st.button("Previous"):
            st.session_state.group_index = (st.session_state.group_index - 1) % len(pending_groups)
            st.experimental_rerun()
    with col_skip:
        if st.button("Skip group"):
            st.session_state.group_index += 1
            st.experimental_rerun()
    with col_next:
        if st.button("Next (no save)"):
            st.session_state.group_index += 1
            st.experimental_rerun()


if __name__ == "__main__":
    main()

