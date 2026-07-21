import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "script" / "exp7_17"
sys.path.insert(0, str(SCRIPT_DIR))

from talr_analysis_common import (  # noqa: E402
    bootstrap_delta,
    extract_prediction,
    mcnemar_exact,
    paired_groups,
    score_row,
)


def row(sample_id, gold, prediction):
    return {
        "id": sample_id,
        "answer": gold,
        "options": "(A) first\n(B) second",
        "model_answer": prediction,
        "error_type": None,
    }


def test_last_answer_has_priority_and_failed_extraction_is_none():
    assert extract_prediction("Answer: A\nAfter checking, final answer: B") == "B"
    assert extract_prediction("No explicit choice here") is None
    assert score_row(row(1, "A", "No explicit choice here"))[
        "failed_extraction"
    ]


def test_pairwise_groups_are_mutually_exclusive():
    reference = {
        "1": row(1, "A", "Answer: B"),
        "2": row(2, "A", "Answer: A"),
        "3": row(3, "B", "Answer: B"),
        "4": row(4, "B", "Answer: A"),
    }
    method = {
        "1": row(1, "A", "Answer: A"),
        "2": row(2, "A", "Answer: B"),
        "3": row(3, "B", "Answer: B"),
        "4": row(4, "B", "Answer: A"),
    }
    groups = paired_groups(reference, method)
    assert groups["fixed"] == ["1"]
    assert groups["damaged"] == ["2"]
    assert groups["both_correct"] == ["3"]
    assert groups["both_wrong"] == ["4"]
    assert sum(len(values) for values in groups.values()) == 4


def test_statistics_are_deterministic():
    assert mcnemar_exact(4, 0) == 0.125
    first = bootstrap_delta([False, True], [True, True], draws=100)
    second = bootstrap_delta([False, True], [True, True], draws=100)
    assert first == second
