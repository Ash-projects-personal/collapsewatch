import json
import subprocess
import sys
from pathlib import Path

import pytest

from collapsewatch.fix import fix_rows
from collapsewatch.io import RowError, read_rows
from collapsewatch.metrics import measure
from collapsewatch.score import score_measurements
from collapsewatch.text import jaccard, ngrams, template_key

ROOT = Path(__file__).parent.parent


def small_rows():
    return [
        {"prompt": "What is photosynthesis?", "response": "Plants turn sunlight into food using chlorophyll in their leaves."},
        {"prompt": "What is photosynthesis?", "response": "Plants turn sunlight into food using chlorophyll in their leaves."},
        {"prompt": "What is photosynthesis?", "response": "Plants turn sunlight into food using chlorophyll in their leaves and oxygen too."},
        {"prompt": "How do bees make honey?", "response": "Bees collect nectar from flowers and fan it with their wings until it thickens into honey."},
        {"prompt": "Summarize the water cycle.", "response": "Water evaporates from oceans, condenses into clouds, and falls back as rain."},
    ]


def test_exact_duplicate_rate_counts_repeated_pairs():
    m = measure(small_rows())
    assert m["rows"] == 5
    assert m["exact_dup_rows"] == 1
    assert m["exact_dup_rate"] == 0.2


def test_template_key_groups_same_prefix_and_masks_numbers():
    assert template_key("Write a product description for the Acme Widget") == template_key(
        "Write a product description for the Bolt Mug"
    )
    assert template_key("Summarize article 12 about bees") == template_key(
        "Summarize article 99 about bees"
    )
    assert template_key("What is photosynthesis?") != template_key(
        "How do bees make honey?"
    )
    # Identifiers that contain a digit are not numbers.
    assert template_key("Use p1 in the answer") != template_key("Use p2 in the answer")


def test_jaccard_is_one_for_identical_and_zero_for_disjoint():
    assert jaccard(ngrams("the quick brown fox jumps over"), ngrams("the quick brown fox jumps over")) == 1.0
    assert jaccard(ngrams("alpha beta gamma delta epsilon"), ngrams("one two three four five")) == 0.0
    assert jaccard(set(), set()) == 0.0


def test_near_duplicate_counts_close_paraphrase_not_different_text():
    rows = [
        {"prompt": "p1", "response": "The Acme Widget is a great choice for everyday use with a sturdy build and fair price today"},
        {"prompt": "p2", "response": "The Acme Widget is a great choice for everyday use with a sturdy build and fair price now"},
        {"prompt": "p3", "response": "Quantum computers use qubits that hold many states at once through superposition and entanglement"},
    ]
    m = measure(rows)
    assert m["near_dup_rows"] == 2
    assert m["near_dup_rate"] == pytest.approx(2 / 3, abs=0.01)


def test_score_is_bounded_and_verdict_bands_hold():
    low = score_measurements(
        {"exact_dup_rate": 0.0, "near_dup_rate": 0.0, "top1_template_share": 0.05,
         "top3_template_share": 0.15, "vocab_top50_coverage": 0.3,
         "length_modal_share": 0.2, "answer_repetition": 0.05}
    )
    assert low["score"] < 35
    assert low["verdict"] == "PASS"
    high = score_measurements(
        {"exact_dup_rate": 0.2, "near_dup_rate": 0.8, "top1_template_share": 0.35,
         "top3_template_share": 0.8, "vocab_top50_coverage": 0.75,
         "length_modal_share": 0.7, "answer_repetition": 0.85}
    )
    assert high["score"] >= 50
    assert high["verdict"] == "FAIL"
    assert 0 <= low["score"] <= 100
    assert 0 <= high["score"] <= 100


def test_fix_removes_exact_and_near_duplicates():
    rows = [
        {"prompt": "p1", "response": "The Acme Widget is a great choice for everyday use with a sturdy build and fair price today"},
        {"prompt": "p1", "response": "The Acme Widget is a great choice for everyday use with a sturdy build and fair price today"},
        {"prompt": "p2", "response": "The Acme Widget is a great choice for everyday use with a sturdy build and fair price now"},
        {"prompt": "p3", "response": "Quantum computers use qubits that hold many states at once through superposition and entanglement"},
    ]
    fixed, stats = fix_rows(rows, max_template_share=1.0)
    assert stats["dropped_exact_duplicates"] == 1
    assert stats["dropped_near_duplicates"] == 1
    assert stats["dropped_template_cap"] == 0
    assert len(fixed) == 2


def test_fix_caps_a_dominant_template():
    rows = [
        {"prompt": "Write a product description for item %d." % i,
         "response": "This is a completely different answer about topic number %d with its own words and facts." % i}
        for i in range(20)
    ]
    rows.append({"prompt": "What is the capital of France?", "response": "Paris is the capital of France."})
    fixed, stats = fix_rows(rows, max_template_share=0.10)
    assert stats["output_rows"] < len(rows)
    assert stats["dropped_template_cap"] > 0


def test_read_rows_rejects_a_bad_row_with_line_number(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"prompt": "ok", "response": "fine"}\n{"prompt": "no response here"}\n'
    )
    with pytest.raises(RowError, match="line 2"):
        read_rows(bad)


def test_read_rows_rejects_an_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(RowError, match="empty"):
        read_rows(empty)


def test_cli_check_exits_zero_on_healthy_and_two_on_collapsed():
    healthy = subprocess.run(
        [sys.executable, "-m", "collapsewatch", "check", str(ROOT / "data" / "healthy.jsonl")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert healthy.returncode == 0, healthy.stdout + healthy.stderr
    collapsed = subprocess.run(
        [sys.executable, "-m", "collapsewatch", "check", str(ROOT / "data" / "collapsed.jsonl")],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert collapsed.returncode == 2, collapsed.stdout + collapsed.stderr


def test_demo_files_are_the_expected_size():
    healthy = read_rows(ROOT / "data" / "healthy.jsonl")
    collapsed = read_rows(ROOT / "data" / "collapsed.jsonl")
    assert len(healthy) == 400
    assert len(collapsed) == 400
    m_healthy = measure(healthy)
    m_collapsed = measure(collapsed)
    assert score_measurements(m_healthy)["verdict"] == "PASS"
    assert score_measurements(m_collapsed)["verdict"] == "FAIL"
    assert score_measurements(m_healthy)["score"] < score_measurements(m_collapsed)["score"]
