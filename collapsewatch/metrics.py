"""Collapse measurements for one fine-tuning JSONL file.

Every number here is a measurement on the file, not a verdict. The
score in score.py turns these into a single risk number; the raw
measurements are what a human should read first.

Measurements
------------
exact_dup_rate      share of rows whose normalized (prompt, response)
                    pair matches an earlier row.
near_dup_rate       share of rows whose response shares a 5-gram
                    Jaccard similarity of 0.8 or more with another
                    row's response. Candidate pairs are found through
                    a 5-gram inverted index, so no pair above the
                    threshold is missed. Exact duplicates also count
                    here (their Jaccard is 1.0); the two rates are
                    reported separately on purpose.
top1_template_share share of rows in the single largest prompt
                    template cluster (first 4 normalized prompt
                    words, digits masked).
top3_template_share share of rows in the three largest clusters.
vocab_top50         share of all response tokens covered by the 50
                    most common response words. High means a narrow
                    vocabulary doing all the work.
length_modal_share  share of rows whose response token count falls in
                    the most common 10-token length bucket.
answer_repetition   size-weighted mean, over template clusters with
                    at least 3 rows, of the mean pairwise 5-gram
                    Jaccard similarity of responses inside the
                    cluster. 0 means every answer in a template is
                    different; 1 means every answer is the same
                    text. Pairs are capped at 200 per cluster,
                    taken in row order, so the cost stays bounded.
type_token_ratio    distinct response tokens / all response tokens.
"""

from collections import Counter, defaultdict
from itertools import combinations

from .text import jaccard, ngrams, template_key, tokens

NEAR_DUP_THRESHOLD = 0.8
MIN_CLUSTER_FOR_ANSWER_REP = 3
MAX_PAIRS_PER_CLUSTER = 200
LENGTH_BUCKET = 10


def _length_stats(lengths):
    if not lengths:
        return {"mean": 0.0, "median": 0, "min": 0, "max": 0}
    ordered = sorted(lengths)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {
        "mean": round(sum(lengths) / n, 1),
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _modal_bucket_share(lengths):
    buckets = Counter(length // LENGTH_BUCKET for length in lengths)
    top = buckets.most_common(1)[0]
    return top[1] / len(lengths), top[0]


def _near_dup_rows(response_shingles):
    """Row indexes that have at least one near-duplicate partner.

    Builds a 5-gram -> row index map, then computes Jaccard only for
    pairs that share at least one 5-gram. Any pair with Jaccard >=
    threshold shares a 5-gram, so this finds every such pair.
    """
    index = defaultdict(set)
    for i, shingles in enumerate(response_shingles):
        for gram in shingles:
            index[gram].add(i)
    flagged = set()
    seen_pairs = set()
    for i, shingles in enumerate(response_shingles):
        if not shingles:
            continue
        partners = set()
        for gram in shingles:
            partners |= index[gram]
        partners.discard(i)
        for j in sorted(partners):
            if j < i:
                continue
            pair = (i, j)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if jaccard(shingles, response_shingles[j]) >= NEAR_DUP_THRESHOLD:
                flagged.add(i)
                flagged.add(j)
    return flagged


def _answer_repetition(clusters, response_shingles):
    """Weighted mean pairwise response similarity inside clusters."""
    weighted_sum = 0.0
    weighted_n = 0
    per_cluster = {}
    for key, idxs in clusters.items():
        if len(idxs) < MIN_CLUSTER_FOR_ANSWER_REP:
            continue
        pairs = list(combinations(sorted(idxs), 2))
        if len(pairs) > MAX_PAIRS_PER_CLUSTER:
            # Stride-sample the pairs so the sample spans the whole
            # cluster instead of only the first few rows.
            stride = -(-len(pairs) // MAX_PAIRS_PER_CLUSTER)
            pairs = pairs[::stride][:MAX_PAIRS_PER_CLUSTER]
        sims = [
            jaccard(response_shingles[i], response_shingles[j]) for i, j in pairs
        ]
        mean_sim = sum(sims) / len(sims) if sims else 0.0
        per_cluster[key] = round(mean_sim, 3)
        weighted_sum += mean_sim * len(idxs)
        weighted_n += len(idxs)
    overall = weighted_sum / weighted_n if weighted_n else 0.0
    return round(overall, 3), per_cluster


def measure(rows):
    """Measure a list of {"prompt", "response"} rows. Returns a dict."""
    n = len(rows)
    prompts = [r["prompt"] for r in rows]
    responses = [r["response"] for r in rows]

    # Exact duplicates on the normalized pair.
    seen = set()
    dup_rows = 0
    for prompt, response in zip(prompts, responses):
        key = (prompt.lower().strip(), response.lower().strip())
        if key in seen:
            dup_rows += 1
        else:
            seen.add(key)

    response_shingles = [ngrams(r) for r in responses]
    near_rows = _near_dup_rows(response_shingles)

    # Template clusters from prompt prefixes.
    clusters = defaultdict(list)
    for i, prompt in enumerate(prompts):
        clusters[template_key(prompt)].append(i)
    sizes = sorted(
        ((key, len(idxs)) for key, idxs in clusters.items()),
        key=lambda kv: -kv[1],
    )
    top1 = sizes[0][1] / n if sizes else 0.0
    top3 = sum(size for _, size in sizes[:3]) / n if sizes else 0.0

    # Vocabulary and lengths.
    resp_tokens = [tokens(r) for r in responses]
    all_resp_tokens = [tok for toks in resp_tokens for tok in toks]
    token_counter = Counter(all_resp_tokens)
    total_tokens = len(all_resp_tokens)
    top50_covered = (
        sum(count for _, count in token_counter.most_common(50)) / total_tokens
        if total_tokens
        else 0.0
    )
    ttr = len(token_counter) / total_tokens if total_tokens else 0.0
    resp_lengths = [len(toks) for toks in resp_tokens]
    modal_share, modal_bucket = _modal_bucket_share(resp_lengths)

    answer_rep, per_cluster_rep = _answer_repetition(clusters, response_shingles)

    # Worst clusters: largest first, with an example prompt and how
    # repetitive their answers are.
    worst = []
    for key, size in sizes[:5]:
        idxs = clusters[key]
        distinct_answers = len({responses[i].lower().strip() for i in idxs})
        worst.append(
            {
                "template": key,
                "rows": size,
                "share": round(size / n, 3),
                "distinct_answers": distinct_answers,
                "mean_pairwise_response_similarity": per_cluster_rep.get(key),
                "example_prompt": prompts[idxs[0]],
            }
        )

    return {
        "rows": n,
        "exact_dup_rows": dup_rows,
        "exact_dup_rate": round(dup_rows / n, 3),
        "near_dup_rows": len(near_rows),
        "near_dup_rate": round(len(near_rows) / n, 3),
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "template_clusters": len(clusters),
        "top1_template_share": round(top1, 3),
        "top3_template_share": round(top3, 3),
        "vocab_top50_coverage": round(top50_covered, 3),
        "type_token_ratio": round(ttr, 3),
        "response_length": _length_stats(resp_lengths),
        "length_modal_share": round(modal_share, 3),
        "length_modal_bucket_tokens": "%d-%d"
        % (modal_bucket * LENGTH_BUCKET, modal_bucket * LENGTH_BUCKET + 9),
        "answer_repetition": answer_rep,
        "worst_templates": worst,
    }
