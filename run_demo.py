"""Demo: check the healthy file, check the collapsed file, fix the
collapsed file, check the fixed file.

Run from the project root:
  python3 run_demo.py

It writes runs/report.json and prints a table. No API key, no
network, no downloads. All numbers are measured by running the
checker on the demo files in data/.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def run_check(path, threshold=50):
    proc = subprocess.run(
        [sys.executable, "-m", "collapsewatch", "check", str(path),
         "--threshold", str(threshold)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc


def parse_score(output):
    for line in output.splitlines():
        if line.startswith("collapse risk score:"):
            return float(line.split(":")[1].split("/")[0].strip())
    return None


def main():
    started = time.time()
    healthy = run_check(ROOT / "data" / "healthy.jsonl")
    collapsed = run_check(ROOT / "data" / "collapsed.jsonl")
    fix = subprocess.run(
        [sys.executable, "-m", "collapsewatch", "fix",
         str(ROOT / "data" / "collapsed.jsonl"),
         str(ROOT / "data" / "fixed.jsonl")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    fixed = run_check(ROOT / "data" / "fixed.jsonl")

    report = {
        "seconds": round(time.time() - started, 2),
        "healthy": {
            "exit_code": healthy.returncode,
            "score": parse_score(healthy.stdout),
            "stdout": healthy.stdout,
        },
        "collapsed": {
            "exit_code": collapsed.returncode,
            "score": parse_score(collapsed.stdout),
            "stdout": collapsed.stdout,
        },
        "fix": {"exit_code": fix.returncode, "stdout": fix.stdout},
        "fixed": {
            "exit_code": fixed.returncode,
            "score": parse_score(fixed.stdout),
            "stdout": fixed.stdout,
        },
    }
    out = ROOT / "runs" / "report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("CollapseWatch demo")
    print("")
    for name in ("healthy", "collapsed", "fixed"):
        item = report[name]
        print("%s: score %s, exit code %d" % (name, item["score"], item["exit_code"]))
    print("")
    print(fix.stdout, end="")
    print("wrote %s in %.1f seconds" % (out, report["seconds"]))
    return report


if __name__ == "__main__":
    main()
