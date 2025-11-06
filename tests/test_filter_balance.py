from __future__ import annotations

from scripts import filter_balance


def build_record(**overrides):
    base = {
        "question": "A block slides on a smooth plane with constant speed.",
        "options": [
            "The resultant force is zero.",
            "The block accelerates downhill.",
            "Friction acts upwards.",
            "The normal contact force is zero.",
        ],
        "correct_index": 0,
        "hint": "Consider Newton's first law.",
        "explanation": "With zero resultant force the block continues at constant velocity.",
        "ao": "AO1+AO2",
        "topic": "Mechanics > Dynamics",
        "difficulty": "AS",
        "sources": [
            {"type": "question_part", "question_id": "TST/01", "part_code": "01.1"},
            {"type": "mark_scheme", "question_id": "TST/01", "part_code": "01.1"},
        ],
    }
    base.update(overrides)
    return base


def test_run_filter_normalises_difficulty():
    record = build_record(difficulty="easy")
    summary = filter_balance.run_filter([record])
    assert len(summary.kept) == 1
    assert summary.kept[0]["difficulty"] == "Easy"


def test_run_filter_flags_missing_difficulty():
    record = build_record(difficulty="")
    summary = filter_balance.run_filter([record])
    assert len(summary.kept) == 0
    assert len(summary.flagged) == 1
    assert "difficulty_missing" in summary.flagged[0]["errors"]


def test_run_filter_detects_duplicate_question():
    record_one = build_record()
    record_two = build_record(question="A block slides on a smooth plane with constant speed.")
    summary = filter_balance.run_filter([record_one, record_two])
    assert len(summary.kept) == 1
    assert len(summary.flagged) == 1
    assert any("duplicate" in err for err in summary.flagged[0]["errors"])


def test_strict_mode_drops_warning_only_records():
    record = build_record(topic="Physics > Misc")
    summary = filter_balance.run_filter([record], strict=True)
    assert len(summary.kept) == 0
    assert len(summary.flagged) == 1
    assert "topic_generic" in summary.flagged[0]["warnings"]


