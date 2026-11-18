#!/usr/bin/env python3
"""Select one best hard-audited trajectory per sample."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TRACE_DIR = Path("/data/keenSQL/train_prompt/sft_privileged_traces")
sys.path.insert(0, str(TRACE_DIR))

import build_sft_traces as B  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--audits", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selected-audit", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) != len(args.audits):
        raise ValueError("--inputs and --audits must have the same length")

    grouped: dict[str, list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], str]]] = {}
    source_candidates: Counter[str] = Counter()
    source_hard_pass: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    hard_issue_counts: Counter[str] = Counter()
    total = 0
    for input_index, (input_path, audit_path) in enumerate(zip(args.inputs, args.audits)):
        rows = load_jsonl(input_path)
        audits = load_jsonl(audit_path)
        if len(rows) != len(audits):
            raise ValueError(f"row count mismatch: {input_path} vs {audit_path}")
        source = str(input_path)
        for row_index, (row, audit) in enumerate(zip(rows, audits), 1):
            total += 1
            if row.get("id") != audit.get("id"):
                raise ValueError(f"row id mismatch at {input_path}:{row_index}")
            sample_id = str((row.get("metadata") or {}).get("sample_id") or "")
            source_candidates[source] += 1
            if not audit.get("hard_pass"):
                hard_issue_counts.update(audit.get("issues") or [])
                continue
            source_hard_pass[source] += 1
            warnings = list(audit.get("warnings") or [])
            warning_counts.update(warnings)
            warning_rank = (
                int("possibly_irrelevant_diagnostic_tool_call" in warnings),
                int("temporally_unseen_gold_numeric_literal" in warnings),
                int("verified_call_is_not_last_tool_call" in warnings),
                len(warnings),
            )
            quality = warning_rank + B.merge_quality_key(row, input_index, row_index)
            grouped.setdefault(sample_id, []).append((quality, row, audit, source))

    selected_rows: list[dict[str, Any]] = []
    selected_audits: list[dict[str, Any]] = []
    selected_by_source: Counter[str] = Counter()
    selected_warning_counts: Counter[str] = Counter()
    alternatives = 0
    for sample_id in sorted(grouped):
        candidates = grouped[sample_id]
        quality, row, audit, source = min(candidates, key=lambda item: item[0])
        selected_rows.append(row)
        selected_audits.append(audit)
        selected_by_source[source] += 1
        selected_warning_counts.update(audit.get("warnings") or [])
        alternatives += len(candidates) - 1

    write_jsonl(args.out, selected_rows)
    write_jsonl(args.selected_audit, selected_audits)
    summary = {
        "selection_policy": "one_best_hard_audited_trajectory_per_sample_id",
        "total_candidates": total,
        "hard_pass_candidates": sum(source_hard_pass.values()),
        "hard_fail_candidates": total - sum(source_hard_pass.values()),
        "selected_records": len(selected_rows),
        "unique_sample_ids": len(grouped),
        "hard_pass_alternatives_dropped": alternatives,
        "candidates_by_source": dict(source_candidates),
        "hard_pass_by_source": dict(source_hard_pass),
        "selected_by_source": dict(selected_by_source),
        "all_candidate_warning_counts": dict(warning_counts),
        "selected_warning_counts": dict(selected_warning_counts),
        "hard_issue_counts": dict(hard_issue_counts),
        "output": str(args.out),
        "selected_audit": str(args.selected_audit),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
