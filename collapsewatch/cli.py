"""CollapseWatch CLI.

  python3 -m collapsewatch check FILE.jsonl [--threshold 50] [--json OUT.json]
  python3 -m collapsewatch fix IN.jsonl OUT.jsonl [--max-template-share 0.10]

Exit codes for `check`: 0 when the score is below the threshold, 2
when it reaches it. 3 for an unreadable or malformed input file.
"""

import argparse
import sys

from . import __version__
from .fix import fix_file
from .io import RowError, read_rows
from .metrics import measure
from .report import json_report, text_report, write_json_report
from .score import DEFAULT_THRESHOLD, score_measurements


def cmd_check(args):
    try:
        rows = read_rows(args.file)
    except (RowError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 3
    measurements = measure(rows)
    scored = score_measurements(measurements)
    print(text_report(args.file, measurements, scored), end="")
    if args.json:
        write_json_report(args.json, json_report(args.file, measurements, scored))
        print("wrote %s" % args.json)
    if scored["score"] >= args.threshold:
        print(
            "score %.1f reaches the threshold %.1f: failing"
            % (scored["score"], args.threshold)
        )
        return 2
    return 0


def cmd_fix(args):
    try:
        rows = read_rows(args.file)
    except (RowError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 3
    fixed, stats = fix_file(args.file, args.output, rows, args.max_template_share)
    print("input rows: %d" % stats["input_rows"])
    print("dropped exact duplicates: %d" % stats["dropped_exact_duplicates"])
    print("dropped near duplicates: %d" % stats["dropped_near_duplicates"])
    print(
        "dropped by template cap (max %.0f%% per template, cap %d rows): %d"
        % (
            stats["max_template_share"] * 100,
            stats["template_cap_rows"],
            stats["dropped_template_cap"],
        )
    )
    print("output rows: %d" % stats["output_rows"])
    print("wrote %s" % args.output)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="collapsewatch",
        description="Pre-training gate for synthetic fine-tuning data.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="score a JSONL file for collapse")
    check.add_argument("file", help="fine-tuning JSONL file")
    check.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="fail when score reaches this (default %(default)s)",
    )
    check.add_argument("--json", help="write the full report as JSON to this path")
    check.set_defaults(func=cmd_check)

    fix = sub.add_parser("fix", help="dedupe a file and cap template concentration")
    fix.add_argument("file", help="input JSONL file")
    fix.add_argument("output", help="output JSONL file")
    fix.add_argument(
        "--max-template-share",
        type=float,
        default=0.10,
        help="no template may exceed this share of the file (default %(default)s)",
    )
    fix.set_defaults(func=cmd_fix)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
