#!/usr/bin/env python3
"""Collect report statistics for the hard-audited (non-semantic) SFT merge."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, "/data/keenSQL/train_prompt/sft_privileged_traces")
import build_sft_traces as B  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dataset_name(question_id: Any) -> str:
    value = str(question_id or "")
    if value.startswith("bird_"):
        return "bird_train"
    if value.isdigit():
        return "bird-verified-2.5k"
    if value.startswith("omni_"):
        return "omni-2k"
    if value.startswith("spider_"):
        return "spider1"
    return "unknown"


def distribution(rows: list[dict[str, Any]], originals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cross: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source = row
        if originals is not None:
            index = int((row.get("metadata") or {}).get("source_index", -1))
            source = originals[index]
        cross[dataset_name(source.get("question_id"))][str(source.get("difficulty") or "unknown")] += 1
    datasets = ("bird_train", "bird-verified-2.5k", "omni-2k", "spider1", "unknown")
    difficulties = ("simple", "moderate", "challenging", "unknown")
    result: dict[str, Any] = {}
    for dataset in datasets:
        if not cross[dataset]:
            continue
        result[dataset] = {name: cross[dataset][name] for name in difficulties}
        result[dataset]["total"] = sum(cross[dataset].values())
    totals = Counter()
    for values in cross.values():
        totals.update(values)
    result["ALL"] = {name: totals[name] for name in difficulties}
    result["ALL"]["total"] = sum(totals.values())
    return result


def final_sql(row: dict[str, Any], limit: int = 520) -> str:
    start = int(row["prompt_messages"])
    for message in reversed(row["messages"][start:]):
        if message.get("role") != "assistant":
            continue
        try:
            parsed = B.T.parse_assistant_turn(
                str(message.get("content", "")),
                require_final_scratch_pad=True,
                require_tool_scratch_pad=True,
                repair_unclosed_function=False,
            )
        except Exception:
            continue
        if parsed.get("action") == "final_answer":
            sql = re.sub(r"\s+", " ", str(parsed.get("sql") or "")).strip()
            return sql if len(sql) <= limit else sql[: limit - 1] + "…"
    return ""


def source_record(row: dict[str, Any], originals: list[dict[str, Any]]) -> dict[str, Any]:
    index = int((row.get("metadata") or {}).get("source_index", -1))
    return originals[index]


def source_label(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return f"{metadata.get('model')} / {metadata.get('decoding')} / attempt {metadata.get('attempt')}"


def success_examples(
    rows: list[dict[str, Any]], audits: list[dict[str, Any]], originals: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    choices: list[tuple[tuple[int, int, int], dict[str, Any], dict[str, Any]]] = []
    difficulty_order = {"simple": 0, "moderate": 1, "challenging": 2}
    dataset_order = {"bird_train": 0, "bird-verified-2.5k": 1, "omni-2k": 2, "spider1": 3}
    for row, audit in zip(rows, audits):
        source = source_record(row, originals)
        difficulty = str(source.get("difficulty"))
        dataset = dataset_name(source.get("question_id"))
        choices.append((
            (
                difficulty_order.get(difficulty, 9),
                dataset_order.get(dataset, 9),
                len(final_sql(row)),
            ),
            row,
            audit,
        ))
    result: list[dict[str, Any]] = []
    used_difficulty: set[str] = set()
    used_dataset: set[str] = set()
    for _rank, row, audit in sorted(choices, key=lambda item: item[0]):
        source = source_record(row, originals)
        difficulty = str(source.get("difficulty"))
        dataset = dataset_name(source.get("question_id"))
        if difficulty in used_difficulty or dataset in used_dataset or audit.get("warnings"):
            continue
        evidence = audit.get("evidence") or {}
        result.append({
            "sample_id": (row.get("metadata") or {}).get("sample_id"),
            "question_id": source.get("question_id"),
            "dataset": dataset,
            "difficulty": difficulty,
            "source": source_label(row),
            "tool_calls": evidence.get("tool_calls"),
            "verified_call": evidence.get("verified_call"),
            "question": source.get("question"),
            "final_sql": final_sql(row),
            "format_repairs": (row.get("metadata") or {}).get("format_repairs") or [],
            "replay_grade": evidence.get("replay_grade"),
        })
        used_difficulty.add(difficulty)
        used_dataset.add(dataset)
        if len(result) == 3:
            break
    return result


def failure_example(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    originals: list[dict[str, Any]],
    predicate,
) -> dict[str, Any] | None:
    for row, audit in pairs:
        if not predicate(audit.get("issues") or []):
            continue
        source = source_record(row, originals)
        evidence = audit.get("evidence") or {}
        return {
            "sample_id": (row.get("metadata") or {}).get("sample_id"),
            "question_id": source.get("question_id"),
            "dataset": dataset_name(source.get("question_id")),
            "difficulty": source.get("difficulty"),
            "source": source_label(row),
            "question": source.get("question"),
            "issues": audit.get("issues") or [],
            "literal_hits": evidence.get("temporal_gold_string_hits") or [],
            "observation_mismatches": evidence.get("observation_replay_mismatches") or [],
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--audits", nargs="+", type=Path, required=True)
    parser.add_argument("--repair-manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--merged-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    originals = json.loads(args.original.read_text(encoding="utf-8"))
    all_rows: list[dict[str, Any]] = []
    all_audits: list[dict[str, Any]] = []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    source_counts: dict[str, Any] = {}
    for input_path, audit_path in zip(args.inputs, args.audits):
        rows = load_jsonl(input_path)
        audits = load_jsonl(audit_path)
        if len(rows) != len(audits):
            raise ValueError(f"alignment mismatch: {input_path}")
        all_rows.extend(rows)
        all_audits.extend(audits)
        pairs.extend(zip(rows, audits))
        source_counts[input_path.stem] = {
            "candidates": len(rows),
            "hard_pass": sum(audit.get("hard_pass") is True for audit in audits),
            "hard_fail": sum(audit.get("hard_pass") is not True for audit in audits),
            "unique_samples": len({str((row.get("metadata") or {}).get("sample_id")) for row in rows}),
        }

    merged = load_jsonl(args.merged)
    merged_audits = load_jsonl(args.merged_audit)
    if len(merged) != len(merged_audits):
        raise ValueError("merged alignment mismatch")
    if any(audit.get("hard_pass") is not True for audit in merged_audits):
        raise ValueError("merged contains a hard-fail row")

    generated_samples = {str((row.get("metadata") or {}).get("sample_id")) for row in all_rows}
    merged_samples = {str((row.get("metadata") or {}).get("sample_id")) for row in merged}
    issues = Counter()
    warnings = Counter()
    categories = Counter()
    issues_by_sample: dict[str, set[str]] = defaultdict(set)
    for row, audit in pairs:
        row_issues = list(audit.get("issues") or [])
        issues.update(row_issues)
        warnings.update(audit.get("warnings") or [])
        sample_id = str((row.get("metadata") or {}).get("sample_id"))
        issues_by_sample[sample_id].update(row_issues)
        if row_issues:
            if "temporally_unseen_gold_string_literal" in row_issues:
                categories["unseen_gold_string_literal"] += 1
            if "repeated_identical_tool_call" in row_issues:
                categories["repeated_identical_tool_call"] += 1
            if any(value.startswith("observation_replay_mismatch_call_") for value in row_issues):
                categories["observation_replay_mismatch"] += 1
            if any(value.startswith("unbalanced_") for value in row_issues):
                categories["unbalanced_protocol_tags"] += 1

    terminal = Counter()
    for sample_id in generated_samples - merged_samples:
        terminal.update(issues_by_sample[sample_id])

    selected_warnings = Counter()
    warning_records = 0
    for audit in merged_audits:
        row_warnings = list(audit.get("warnings") or [])
        selected_warnings.update(row_warnings)
        warning_records += int(bool(row_warnings))

    repair_ops = Counter()
    repaired_records = 0
    for path in args.repair_manifests:
        manifest = load_jsonl(path)
        repaired_records += len(manifest)
        for item in manifest:
            repair_ops.update(item.get("repairs") or [])

    def substantive_unseen_literal(values: list[str], pair: tuple[dict[str, Any], dict[str, Any]] | None = None) -> bool:
        return "temporally_unseen_gold_string_literal" in values

    unseen_example = None
    for row, audit in pairs:
        if not substantive_unseen_literal(list(audit.get("issues") or [])):
            continue
        hits = (audit.get("evidence") or {}).get("temporal_gold_string_hits") or []
        if not any(
            re.search(r"[A-Za-z]{3}", str(hit.get("literal") or ""))
            and "%" not in str(hit.get("literal") or "")
            and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(hit.get("literal") or ""))
            for hit in hits
        ):
            continue
        unseen_example = failure_example(
            [(row, audit)], originals, lambda values: "temporally_unseen_gold_string_literal" in values
        )
        break

    failures = [
        unseen_example,
        failure_example(pairs, originals, lambda values: "repeated_identical_tool_call" in values),
        failure_example(pairs, originals, lambda values: any(item.startswith("unbalanced_") for item in values)),
    ]
    payload = {
        "original_distribution": distribution(originals),
        "merged_distribution": distribution(merged, originals),
        "pipeline": {
            "original_hard_samples": len(originals),
            "ex_correct_candidate_rows": len(all_rows),
            "samples_with_ex_correct_candidate": len(generated_samples),
            "samples_without_ex_correct_candidate": len(originals) - len(generated_samples),
            "hard_pass_candidate_rows": sum(audit.get("hard_pass") is True for audit in all_audits),
            "hard_fail_candidate_rows": sum(audit.get("hard_pass") is not True for audit in all_audits),
            "merged_records": len(merged),
            "samples_lost_because_all_candidates_failed_hard_audit": len(generated_samples - merged_samples),
            "valid_alternatives_dropped": sum(audit.get("hard_pass") is True for audit in all_audits) - len(merged),
        },
        "source_counts": source_counts,
        "format_repairs": {
            "repaired_candidate_records": repaired_records,
            "operation_counts": dict(repair_ops),
            "selected_repaired_records": sum(bool((row.get("metadata") or {}).get("format_repairs")) for row in merged),
        },
        "hard_failures": {
            "category_record_counts_overlapping": dict(categories),
            "raw_issue_counts_overlapping": dict(issues),
            "terminal_sample_issue_counts_overlapping": dict(terminal),
        },
        "merged_nonfatal_warnings": {
            "records_with_any_warning": warning_records,
            "warning_counts_overlapping": dict(selected_warnings),
        },
        "success_examples": success_examples(merged, merged_audits, originals),
        "failure_examples": [item for item in failures if item is not None],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
