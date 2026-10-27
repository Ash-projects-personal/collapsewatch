"""Collapse risk score: one number from the measurements, with the
weights written down so anyone can argue with them.

Score 0-100, higher means more collapsed. Each component is a rate
in 0-1, multiplied by 100, multiplied by its weight:

  exact_dup_rate        0.20
  near_dup_rate         0.20
  top1_template_share   0.15
  top3_template_share   0.10
  vocab_top50_coverage  0.10
  length_modal_share    0.10
  answer_repetition     0.15

Verdict bands (stated up front, not fitted to any file):
  score < 35   PASS: no collapse signal worth blocking on.
  35 - 49      WARN: some concentration; worth a look before training.
  score >= 50  FAIL: the file is mostly repeats and templates.

The default CI threshold is 50, the FAIL line. `check` exits 2 when
the score reaches the threshold, 0 otherwise. A WARN still exits 0;
the warning is printed, not hidden.
"""

WEIGHTS = {
    "exact_dup_rate": 0.20,
    "near_dup_rate": 0.20,
    "top1_template_share": 0.15,
    "top3_template_share": 0.10,
    "vocab_top50_coverage": 0.10,
    "length_modal_share": 0.10,
    "answer_repetition": 0.15,
}

PASS_BELOW = 35
FAIL_AT = 50
DEFAULT_THRESHOLD = 50


def score_measurements(m):
    """Score a measurement dict. Returns {"score", "components", "verdict"}."""
    components = {}
    total = 0.0
    for key, weight in WEIGHTS.items():
        value = float(m[key]) * 100.0 * weight
        components[key] = round(value, 1)
        total += value
    score = round(total, 1)
    if score < PASS_BELOW:
        verdict = "PASS"
    elif score < FAIL_AT:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    return {"score": score, "components": components, "verdict": verdict}
