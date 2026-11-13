"""Identify raw S3 scores produced after an exhausted empty-response call."""
from datetime import datetime
from pathlib import Path
import re

START = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3}[^\n]*\[Example ID\]: (\S+)")
EXHAUSTED = re.compile(r"Attempt 3 failed: [^\n]*empty_completion[^\n]*\nMax retries reached\. Handling failure\.")

def invalid_scores(worker):
    log = worker / "worker.log"
    if not log.is_file():
        return set()
    text = log.read_text(errors="replace")
    by_task = {}
    boundaries = list(START.finditer(text))
    for idx, match in enumerate(boundaries):
        end = boundaries[idx+1].start() if idx+1 < len(boundaries) else len(text)
        when = datetime.fromisoformat(match[1]).timestamp()
        by_task.setdefault(match[2], []).append((when, bool(EXHAUSTED.search(text, match.end(), end))))
    invalid = set()
    for p in (worker / "results").rglob("result.txt"):
        if "attempts" in p.parts:
            continue
        scored_at = p.stat().st_mtime
        eligible = [flag for when, flag in by_task.get(p.parent.name, []) if when <= scored_at]
        if eligible and eligible[-1]:
            invalid.add(p)
    return invalid
