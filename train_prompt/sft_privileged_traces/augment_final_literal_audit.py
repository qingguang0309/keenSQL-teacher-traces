#!/usr/bin/env python3
"""Add final-SQL temporal literal checks to an existing aligned hard audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/tmp")
sys.path.insert(0, "/data/keenSQL/train_prompt/sft_privileged_traces")

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


def final_hits(row: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_messages = int(row["prompt_messages"])
    messages = row["messages"]
    visible = "\n".join(
        [
            str(row.get("question") or ""),
            str(row.get("evidence") or ""),
            str(messages[prompt_messages - 1].get("content", "")),
        ]
    )
    strings, _numbers = A.sql_literals(str(row.get("gold_sql") or ""))
    hits: list[dict[str, Any]] = []
    for turn, item in enumerate(messages[prompt_messages:], 1):
        if item.get("role") == "user":
            visible += "\n" + A.unwrap_observation(str(item.get("content", "")))
            continue
        if item.get("role") != "assistant":
            continue
        try:
            parsed = B.T.parse_assistant_turn(
                str(item.get("content", "")),
                require_final_scratch_pad=True,
                require_tool_scratch_pad=True,
                repair_unclosed_function=False,
            )
        except Exception:
            continue
        if parsed.get("action") != "final_answer":
            continue
        final_sql = str(parsed.get("sql") or "")
        for literal in strings:
            if len(literal.strip("%_")) < 2:
                continue
            if A.text_has_literal(final_sql, literal) and not A.text_has_literal(visible, literal):
                hits.append({"turn": turn, "literal": literal, "location": "final_sql"})
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--validated-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    audits = load_jsonl(args.audit)
    if len(rows) != len(audits):
        raise ValueError("row count mismatch")
    hit_records = newly_failed = 0
    hit_literals: dict[str, int] = {}
    for row, audit in zip(rows, audits):
        hits = final_hits(row)
        if not hits:
            continue
        hit_records += 1
        was_pass = audit.get("hard_pass") is True
        evidence = audit.setdefault("evidence", {})
        existing = list(evidence.get("temporal_gold_string_hits") or [])
        evidence["temporal_gold_string_hits"] = existing + hits
        issues = list(audit.get("issues") or [])
        if "temporally_unseen_gold_string_literal" not in issues:
            issues.append("temporally_unseen_gold_string_literal")
        audit["issues"] = issues
        audit["hard_pass"] = False
        newly_failed += int(was_pass)
        for hit in hits:
            literal = str(hit["literal"])
            hit_literals[literal] = hit_literals.get(literal, 0) + 1
    write_jsonl(args.audit, audits)
    write_jsonl(
        args.validated_out,
        [row for row, audit in zip(rows, audits) if audit.get("hard_pass") is True],
    )
    report = {
        "records_with_unseen_gold_literal_in_final_sql": hit_records,
        "newly_failed_records": newly_failed,
        "hard_pass_records_after": sum(audit.get("hard_pass") is True for audit in audits),
        "top_literals": dict(sorted(hit_literals.items(), key=lambda item: (-item[1], item[0]))[:20]),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
