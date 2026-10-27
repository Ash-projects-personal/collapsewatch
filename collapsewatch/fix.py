"""Fix a collapsed file: dedupe it and cap template concentration.

Steps, in order, all deterministic (first occurrence wins):
  1. Drop exact duplicate rows (same normalized prompt + response).
  2. Drop near-duplicate rows: a row is dropped when its response has
     5-gram Jaccard >= 0.8 with a row already kept.
  3. Cap each prompt template cluster at max_template_share of the
     file: rows past the cap are dropped by evenly spaced stride, so
     the surviving rows still span the cluster's range of topics.

This does not invent data. It only removes rows. A file that is
collapsed because every answer is the same skeleton will still show
that after fixing; the score should move, not vanish.
"""

from collections import defaultdict

from .io import write_rows
from .metrics import NEAR_DUP_THRESHOLD
from .text import jaccard, ngrams, template_key


def fix_rows(rows, max_template_share=0.10):
    """Return (kept_rows, stats) after the three fix steps."""
    stats = {"input_rows": len(rows)}

    # Step 1: exact duplicates.
    seen = set()
    kept = []
    dropped_exact = 0
    for row in rows:
        key = (row["prompt"].lower().strip(), row["response"].lower().strip())
        if key in seen:
            dropped_exact += 1
        else:
            seen.add(key)
            kept.append(row)
    stats["dropped_exact_duplicates"] = dropped_exact

    # Step 2: near duplicates, greedy in file order.
    kept_shingles = []
    after_near = []
    dropped_near = 0
    for row in kept:
        shingles = ngrams(row["response"])
        is_dup = any(
            jaccard(shingles, other) >= NEAR_DUP_THRESHOLD for other in kept_shingles
        )
        if is_dup:
            dropped_near += 1
        else:
            kept_shingles.append(shingles)
            after_near.append(row)
    stats["dropped_near_duplicates"] = dropped_near

    # Step 3: cap template clusters.
    clusters = defaultdict(list)
    for i, row in enumerate(after_near):
        clusters[template_key(row["prompt"])].append(i)
    cap = max(1, int(round(max_template_share * len(after_near))))
    drop = set()
    dropped_cap = 0
    for idxs in clusters.values():
        if len(idxs) <= cap:
            continue
        keep = set(idxs[:: max(1, len(idxs) // cap)][:cap])
        for i in idxs:
            if i not in keep:
                drop.add(i)
                dropped_cap += 1
    final = [row for i, row in enumerate(after_near) if i not in drop]
    stats["dropped_template_cap"] = dropped_cap
    stats["max_template_share"] = max_template_share
    stats["template_cap_rows"] = cap
    stats["output_rows"] = len(final)
    return final, stats


def fix_file(input_path, output_path, rows, max_template_share=0.10):
    """Fix rows from an already-read file and write the result."""
    fixed, stats = fix_rows(rows, max_template_share=max_template_share)
    write_rows(output_path, fixed)
    return fixed, stats
