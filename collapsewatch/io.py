"""Reading fine-tuning JSONL files.

Accepted row shapes (one JSON object per line):
  {"prompt": ..., "response": ...}
  {"instruction": ..., "response": ...}
  {"prompt": ..., "completion": ...}
  {"input": ..., "output": ...}

Rows missing both a prompt-side and a response-side field are rejected
with the line number, so a bad file fails loudly instead of scoring
silently on empty text.
"""

import json


class RowError(ValueError):
    """A row in the input file is unusable; carries the line number."""


_PROMPT_KEYS = ("prompt", "instruction", "input")
_RESPONSE_KEYS = ("response", "completion", "output")


def read_rows(path):
    """Read a JSONL file into a list of {"prompt", "response"} dicts.

    Raises RowError on a blank line, a non-JSON line, a non-object
    row, or a row missing a prompt or a response.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                raise RowError("line %d: blank line" % lineno)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RowError("line %d: not valid JSON (%s)" % (lineno, exc))
            if not isinstance(obj, dict):
                raise RowError("line %d: row is not a JSON object" % lineno)
            prompt = next(
                (obj[k] for k in _PROMPT_KEYS if isinstance(obj.get(k), str)),
                None,
            )
            response = next(
                (obj[k] for k in _RESPONSE_KEYS if isinstance(obj.get(k), str)),
                None,
            )
            if prompt is None:
                raise RowError(
                    "line %d: no prompt field (%s)"
                    % (lineno, "/".join(_PROMPT_KEYS))
                )
            if response is None:
                raise RowError(
                    "line %d: no response field (%s)"
                    % (lineno, "/".join(_RESPONSE_KEYS))
                )
            rows.append({"prompt": prompt, "response": response})
    if not rows:
        raise RowError("file is empty: %s" % path)
    return rows


def write_rows(path, rows):
    """Write rows back out as JSONL."""
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
