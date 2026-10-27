# CollapseWatch

A pre-training gate for synthetic fine-tuning data.

## Pitch

Synthetic data is now most of the training data for small fine tunes. A generated dataset can look large and be small in practice. The same prompt template 400 times, the same answer shape, the same few phrases. The model learns the template, not the task. Nobody checks this before the training run starts.

CollapseWatch points at a fine-tuning JSONL file before you train. It measures exact duplicates, near duplicates, template concentration, vocabulary narrowness, length concentration, and answer repetition inside templates. It turns those into one collapse risk score, names the worst templates, and exits non-zero in CI when the score is over the threshold. It can also fix a file: dedupe it and cap each template, then re-score it.

## What it does

It reads a JSONL file of prompt and response pairs. It checks seven things.

Exact duplicates. Rows whose normalized prompt and response match an earlier row.

Near duplicates. Rows whose response shares a 5-gram Jaccard similarity of 0.8 or more with another row. Candidate pairs come from a 5-gram inverted index, so no pair above the threshold is missed.

Template concentration. Prompts are clustered by their first four normalized words, with pure numbers masked. Prompts that share a template collapse to one key. The score uses the largest cluster share and the top three clusters share.

Vocabulary. The share of response tokens covered by the 50 most common response words. High means a narrow vocabulary is doing all the work.

Length concentration. The share of rows whose response token count falls in the most common 10-token length bucket.

Answer repetition. The size-weighted mean, over template clusters with at least 3 rows, of the mean pairwise 5-gram Jaccard similarity of responses inside the cluster. Zero means every answer in a template is different. One means every answer is the same text.

## How the score works

Score 0 to 100. Higher means more collapsed. Each component is a rate from 0 to 1, times 100, times its weight.

| Component | Weight |
|---|---|
| Exact duplicate rate | 0.20 |
| Near duplicate rate | 0.20 |
| Largest template share | 0.15 |
| Top 3 templates share | 0.10 |
| Top-50 word coverage | 0.10 |
| Modal length bucket share | 0.10 |
| Answer repetition | 0.15 |

Verdict bands, stated up front: below 35 is PASS, 35 to 49 is WARN, 50 and above is FAIL. The default CI threshold is 50. `check` exits 2 when the score reaches the threshold, 0 otherwise. A WARN still exits 0. The warning is printed, not hidden.

## How to run

No API key. No network. No downloads. Python 3.10 or newer. Pure standard library, except pytest for the tests.

```
cd collapsewatch
python3 -m collapsewatch check data/healthy.jsonl
python3 -m collapsewatch check data/collapsed.jsonl --threshold 50
python3 -m collapsewatch check file.jsonl --json runs/report.json
python3 -m collapsewatch fix data/collapsed.jsonl data/fixed.jsonl
python3 run_demo.py
```

Tests need pytest only:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```

The demo writes `runs/report.json`. The demo data generator is `data/make_demo_data.py`. Both demo files are invented fixtures with a fixed seed, so regenerating them gives the same files.

## Real numbers from the demo run

Run on 2026-10-27. The demo runs in under 4 seconds. Both files have 400 rows.

| File | Score | Verdict | Exit code |
|---|---|---|---|
| healthy.jsonl | 12.3 | PASS | 0 |
| collapsed.jsonl | 52.8 | FAIL | 2 |
| fixed.jsonl (after `fix`) | 17.1 | PASS | 0 |

Healthy file. 12 prompt templates across 6 task families, 40 topics, answers composed from topic facts plus family sentences. The generator enforces no exact duplicates.

| Measurement | Healthy | Collapsed |
|---|---|---|
| Exact duplicates | 0 rows (0.0%) | 127 rows (31.8%) |
| Near duplicates | 4 rows (1.0%) | 337 rows (84.3%) |
| Distinct prompt templates | 103 | 23 |
| Largest template share | 8.5% | 36.2% |
| Top 3 templates share | 25.0% | 90.0% |
| Top-50 word coverage | 41.1% | 58.0% |
| Type-token ratio | 0.044 | 0.050 |
| Response length, mean / median / range | 59.2 / 59 / 40-88 tokens | 40.9 / 39 / 31-86 tokens |
| Modal length bucket share | 32.3% in 60-69 tokens | 54.0% in 30-39 tokens |
| Answer repetition | 0.067 | 0.261 |

Collapsed file. 320 of 400 rows (80%) come from just 3 prompt templates, with answers built from 2 skeletons per template and a small pool of slot values. 40 rows are exact repeats of earlier rows. The other 40 rows are healthy-style rows.

Worst templates in the collapsed file:

| Template | Rows | Share | Distinct answers | Answer similarity |
|---|---|---|---|---|
| write a product description | 145 | 36.2% | 88 | 0.290 |
| summarize this article about | 124 | 31.0% | 12 | 0.195 |
| write a tweet about | 91 | 22.8% | 12 | 0.359 |

The fix. `collapsewatch fix` on the collapsed file dropped 127 exact duplicates, 150 near duplicates, and 47 rows by the template cap (no template above 10% of the file). 400 rows became 76. The score moved from 52.8 to 17.1. The fix only removes rows. It does not invent data. A file whose answers are all the same skeleton will still show that after fixing. The score should move, not vanish.

Tests: 11 passed in 0.72 seconds.

## Honest limits

N-gram and template checks catch surface collapse, not semantic collapse. Two rows can say the same thing in different words and pass. Say that plainly. It is a gate, not a guarantee.

The template key uses the first four prompt words. A template that puts the variable part in the first four words will split into several clusters, and the concentration will read lower than it is. A template with a very long fixed prefix will cluster correctly.

The near-duplicate threshold of 0.8 on 5-gram Jaccard is a choice, not a law. Short responses have few 5-grams, so one changed word moves their similarity a lot. The threshold and the weights are written down in `collapsewatch/score.py` and `collapsewatch/metrics.py` so anyone can argue with them.

The demo files are invented. The healthy file is built to be healthy and the collapsed file is built to be collapsed. The tool measured them. It did not invent the numbers. Nothing here proves anything about a real fine-tuning run. The point is the gate: run it on your own file before you train, and read the measurements, not just the score.

## Credit

The idea that synthetic data collapses into repeats is the model collapse literature: Shumailov et al., Nature 2024, on recursive training on generated data. The tool does not reimplement the research metrics in that space. GenProof scores pretraining-scale collapse risk, and PLDiv measures diversity with persistence landscapes. CollapseWatch is narrower on purpose: a fast local gate for one fine-tuning file, with thresholds and a CI exit code, and no embeddings and no API calls.

## Name check

CollapseWatch is clean by screening as of 2026-10-27. No GitHub repo, PyPI package, or product named CollapseWatch was found in search on 2026-10-26 or in the re-check on 2026-10-27. A Visual Studio command id called Collapse Watch exists, and a Bluesky handle collapsewatch3 exists. Neither is a product in this space. This is screening, not a legal trademark clearance.
