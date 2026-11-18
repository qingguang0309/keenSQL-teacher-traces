#!/usr/bin/env python3
"""Re-audit only rows whose first replay failed for transient/runtime reasons."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TRACE_DIR = Path("/data/keenSQL/train_prompt/sft_privileged_traces")
sys.path.insert(0, "/tmp")
sys.path.insert(0, str(TRACE_DIR))

import audit_merged_sft as A  # noqa: E402
import build_sft_traces as B  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def transient_issues(audit: dict[str, Any]) -> list[str]:
    issues = list(audit.get("issues") or [])
    transient = [
        issue
        for issue in issues
        if issue.startswith("observation_replay_mismatch_call_")
        or issue.startswith("tool_replay_error_call_")
        or issue == "replayed_grading_infrastructure"
    ]
    grade = (audit.get("evidence") or {}).get("replay_grade") or {}
    if "replayed_ex_not_correct" in issues and any(
        token in str(grade.get("reason") or "")
        for token in ("Timeout", "timed out", "locked", "temporarily")
    ):
        transient.append("replayed_ex_not_correct")
    return transient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--validated-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--db-root", type=Path, default=B.DEFAULT_DB_ROOT)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    audits = load_jsonl(args.audit)
    if len(rows) != len(audits):
        raise ValueError("input/audit row count mismatch")
    targeted = recovered = remained = 0
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    attempts_used: Counter[int] = Counter()
    for index, (row, prior) in enumerate(zip(rows, audits)):
        transient = transient_issues(prior)
        if not transient:
            continue
        targeted += 1
        before.update(transient)
        best = prior
        used = 0
        for attempt in range(1, args.attempts + 1):
            _index, current = A.audit_row(index, row, args.db_root)
            used = attempt
            best = current
            if not transient_issues(current):
                break
        audits[index] = best
        attempts_used[used] += 1
        remaining = transient_issues(best)
        if remaining:
            remained += 1
            after.update(remaining)
        else:
            recovered += 1
        if targeted % 10 == 0:
            print(f"transient re-audit {targeted}", flush=True)

    write_jsonl(args.audit, audits)
    write_jsonl(
        args.validated_out,
        [row for row, audit in zip(rows, audits) if audit.get("hard_pass") is True],
    )
    report = {
        "targeted_records": targeted,
        "recovered_records": recovered,
        "remaining_transient_records": remained,
        "transient_issues_before": dict(before),
        "transient_issues_after": dict(after),
        "attempts_used": dict(attempts_used),
        "hard_pass_records_after": sum(audit.get("hard_pass") is True for audit in audits),
        "hard_fail_records_after": sum(audit.get("hard_pass") is not True for audit in audits),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
