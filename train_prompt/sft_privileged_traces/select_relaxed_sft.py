#!/usr/bin/env python3
"""Select one EX-correct, non-gold-leaking trajectory per sample.

Strict hard-pass candidates remain preferred.  A strict failure may be used only
when final EX is true and it has no temporally unseen gold string literal.
The selected audit preserves the strict result in `strict_*` fields and records
acceptance under the relaxed merge policy separately.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TRACE_DIR = Path("/data/keenSQL/train_prompt/sft_privileged_traces")
sys.path.insert(0, str(TRACE_DIR))

import build_sft_traces as B  # noqa: E402


POLICY = "final_ex_true_without_temporally_unseen_gold_string_literal"
GOLD_ISSUE = "temporally_unseen_gold_string_literal"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def final_ex_true(row: dict[str, Any], audit: dict[str, Any]) -> bool:
    replay = ((audit.get("evidence") or {}).get("replay_grade") or {}).get("ex")
    if replay is True:
        return True
    # Malformed protocol can prevent strict replay even though the generation
    # pipeline already executed and EX-accepted the final SQL.
    return (row.get("metadata") or {}).get("ex") is True


def relaxed_issue_rank(issues: list[str]) -> tuple[int, int, int, int]:
    unbalanced = sum(str(issue).startswith("unbalanced_") for issue in issues)
    observation = sum(str(issue).startswith("observation_replay_mismatch_call_") for issue in issues)
    repeated = int("repeated_identical_tool_call" in issues)
    return (unbalanced, observation, repeated, len(issues))


def selected_row(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(row)
    metadata = value.setdefault("metadata", {})
    strict_issues = list(audit.get("issues") or [])
    metadata["merge_acceptance_policy"] = POLICY
    metadata["strict_hard_pass"] = audit.get("hard_pass") is True
    metadata["relaxed_acceptance_issues"] = [] if audit.get("hard_pass") else strict_issues
    return value


def selected_audit(audit: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(audit)
    strict_pass = audit.get("hard_pass") is True
    strict_issues = list(audit.get("issues") or [])
    value["strict_hard_pass"] = strict_pass
    value["strict_issues"] = strict_issues
    value["merge_pass"] = True
    value["merge_acceptance_policy"] = POLICY
    value["accepted_under_relaxed_policy"] = not strict_pass
    value["relaxed_acceptance_issues"] = [] if strict_pass else strict_issues
    # `hard_pass` and `issues` describe the active merge policy in the final
    # selected audit; the original strict result remains above.
    value["hard_pass"] = True
    value["issues"] = []
    return value


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
    total = 0
    strict_pass_candidates = 0
    relaxed_candidates = 0
    excluded_gold_candidates = 0
    candidates_by_source: Counter[str] = Counter()
    eligible_by_source: Counter[str] = Counter()
    relaxed_issue_counts: Counter[str] = Counter()

    for input_index, (input_path, audit_path) in enumerate(zip(args.inputs, args.audits)):
        rows = load_jsonl(input_path)
        audits = load_jsonl(audit_path)
        if len(rows) != len(audits):
            raise ValueError(f"row count mismatch: {input_path} vs {audit_path}")
        source = str(input_path)
        for row_index, (row, audit) in enumerate(zip(rows, audits), 1):
            total += 1
            candidates_by_source[source] += 1
            if row.get("id") != audit.get("id"):
                raise ValueError(f"row id mismatch at {input_path}:{row_index}")
            issues = list(audit.get("issues") or [])
            if GOLD_ISSUE in issues:
                excluded_gold_candidates += 1
                continue
            if not final_ex_true(row, audit):
                continue

            strict_pass = audit.get("hard_pass") is True
            strict_pass_candidates += int(strict_pass)
            relaxed_candidates += int(not strict_pass)
            if not strict_pass:
                relaxed_issue_counts.update(issues)
            eligible_by_source[source] += 1

            warnings = list(audit.get("warnings") or [])
            warning_rank = (
                int("possibly_irrelevant_diagnostic_tool_call" in warnings),
                int("temporally_unseen_gold_numeric_literal" in warnings),
                int("verified_call_is_not_last_tool_call" in warnings),
                len(warnings),
            )
            quality = (
                int(not strict_pass),
                *relaxed_issue_rank(issues),
                *warning_rank,
                *B.merge_quality_key(row, input_index, row_index),
            )
            sample_id = str((row.get("metadata") or {}).get("sample_id") or "")
            grouped.setdefault(sample_id, []).append((quality, row, audit, source))

    selected_rows: list[dict[str, Any]] = []
    selected_audits: list[dict[str, Any]] = []
    selected_by_source: Counter[str] = Counter()
    selected_relaxed_issues: Counter[str] = Counter()
    relaxed_sample_ids: list[str] = []
    alternatives = 0
    for sample_id in sorted(grouped):
        candidates = grouped[sample_id]
        _quality, row, audit, source = min(candidates, key=lambda item: item[0])
        selected_rows.append(selected_row(row, audit))
        selected_audits.append(selected_audit(audit))
        selected_by_source[source] += 1
        if audit.get("hard_pass") is not True:
            relaxed_sample_ids.append(sample_id)
            selected_relaxed_issues.update(audit.get("issues") or [])
        alternatives += len(candidates) - 1

    write_jsonl(args.out, selected_rows)
    write_jsonl(args.selected_audit, selected_audits)
    summary = {
        "selection_policy": POLICY,
        "total_candidates": total,
        "strict_hard_pass_candidates": strict_pass_candidates,
        "relaxed_eligible_failed_candidates": relaxed_candidates,
        "eligible_candidates": strict_pass_candidates + relaxed_candidates,
        "excluded_gold_literal_candidates": excluded_gold_candidates,
        "selected_records": len(selected_rows),
        "selected_strict_records": len(selected_rows) - len(relaxed_sample_ids),
        "selected_relaxed_records": len(relaxed_sample_ids),
        "selected_relaxed_sample_ids": relaxed_sample_ids,
        "eligible_alternatives_dropped": alternatives,
        "candidates_by_source": dict(candidates_by_source),
        "eligible_by_source": dict(eligible_by_source),
        "selected_by_source": dict(selected_by_source),
        "all_relaxed_issue_counts": dict(relaxed_issue_counts),
        "selected_relaxed_issue_counts": dict(selected_relaxed_issues),
        "output": str(args.out),
        "selected_audit": str(args.selected_audit),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
