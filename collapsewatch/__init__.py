"""CollapseWatch: a pre-training gate for synthetic fine-tuning data.

Point it at a fine-tuning JSONL file before you train. It tells you
whether your synthetic data has already collapsed into repeats, near
duplicates, and a narrow set of templates. Pure Python, no embeddings,
no API calls, no model downloads. Runs anywhere in seconds.

Usage:
  python3 -m collapsewatch check data/healthy.jsonl
  python3 -m collapsewatch check data/collapsed.jsonl --threshold 50
  python3 -m collapsewatch check file.jsonl --json runs/report.json
  python3 -m collapsewatch fix data/collapsed.jsonl data/fixed.jsonl
"""

__version__ = "0.1.0"
