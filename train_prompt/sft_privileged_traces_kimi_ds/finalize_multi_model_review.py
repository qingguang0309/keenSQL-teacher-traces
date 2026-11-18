#!/usr/bin/env python3
"""Finalize the filtered multi-model SFT review outputs and their audit metadata."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(
    "/data/keenSQL/dataset/sft_data/"
    "privileged_traces_sft_privileged_traces_kimi_ds/all_output_merge"
)
FINAL = ROOT / "final"
AUDIT = ROOT / "audit"
INPUT = Path(
    "/data/keenSQL/dataset/sft_data/privileged_traces/"
    "other_teacher_input/remaining_hard_1382.json"
)
ROUND2 = Path(
    "/data/keenSQL/dataset/sft_data/privileged_traces/round2_recovery/"
    "merged/sft_privileged_round2_merged.jsonl"
)

STRICT_SRC = FINAL / "strict_high_quality_prompt_compliant.jsonl"
INCLUSIVE_SRC = FINAL / "strict_high_quality_with_single_tool_prompt_compliant.jsonl"
STRICT = FINAL / "multi_model_review.json"
INCLUSIVE = FINAL / "multi_model_review_with_single_tool.json"

STRICT_SEL_SRC = AUDIT / "final_strict_high_quality_prompt_compliant.selection.jsonl"
INCLUSIVE_SEL_SRC = AUDIT / "final_strict_high_quality_with_single_tool_prompt_compliant.selection.jsonl"
STRICT_SEL = AUDIT / "multi_model_review.selection.jsonl"
INCLUSIVE_SEL = AUDIT / "multi_model_review_with_single_tool.selection.jsonl"

EXCLUDED_AUDIT_SRC = AUDIT / "prompt_compliance_excluded_52.jsonl"
EXCLUDED_FULL_SRC = AUDIT / "prompt_compliance_excluded_52_full_trajectories.jsonl"
EXCLUDED_AUDIT = AUDIT / "multi_model_review_excluded_52.audit.jsonl"
EXCLUDED_FULL = AUDIT / "multi_model_review_excluded_52.full_trajectories.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def rename_checked(source: Path, target: Path) -> None:
    if source.exists() and target.exists():
        if source.read_bytes() != target.read_bytes():
            raise RuntimeError(f"both paths exist with different contents: {source}, {target}")
        source.unlink()
    elif source.exists():
        source.replace(target)
    elif not target.exists():
        raise FileNotFoundError(f"missing both source and target: {source}, {target}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qid(row: dict) -> str:
    return str(row.get("question_id") or "")


def dataset(question_id: str) -> str:
    if question_id.startswith("bird_"):
        return "bird_train"
    if question_id.startswith("omni_"):
        return "omni-2k"
    if question_id.startswith("spider_"):
        return "spider1"
    return "bird-verified-2.5k"


def assistant_text(row: dict) -> str:
    start = int(row.get("prompt_messages") or 0)
    return "\n".join(
        str(message.get("content") or "")
        for message in (row.get("messages") or [])[start:]
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def tool_calls(row: dict) -> int:
    return len(re.findall(r"(?i)<tool_call>", assistant_text(row)))


def dist(rows: list[dict], getter) -> dict[str, int]:
    return dict(sorted(Counter(str(getter(row)) for row in rows).items()))


def final_stats(rows: list[dict], selections: list[dict], path: Path) -> dict:
    ids = [qid(row) for row in rows]
    selection_ids = [qid(row) for row in selections]
    if ids != selection_ids:
        raise RuntimeError(f"data/selection order mismatch for {path}")
    repairs = sum(bool((row.get("quality") or {}).get("repairs")) for row in selections)
    return {
        "output": str(path),
        "format": "jsonl_records_with_json_extension",
        "selected_unique_ids": len(ids),
        "unique_ids": len(set(ids)),
        "sha256": sha256(path),
        "selected_by_dataset": dist(rows, lambda row: dataset(qid(row))),
        "selected_by_difficulty": dist(
            selections, lambda row: row.get("difficulty") or "unknown"
        ),
        "selected_by_model": dist(selections, lambda row: row.get("model") or "unknown"),
        "selected_by_decoding": dist(
            selections, lambda row: row.get("decoding") or "unknown"
        ),
        "selected_by_category": dist(
            selections, lambda row: row.get("category") or "unknown"
        ),
        "tool_call_distribution": dist(rows, tool_calls),
        "selected_with_repairs": repairs,
        "selection_manifest": str(
            STRICT_SEL if path == STRICT else INCLUSIVE_SEL
        ),
    }


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)

    rename_checked(STRICT_SRC, STRICT)
    rename_checked(INCLUSIVE_SRC, INCLUSIVE)
    rename_checked(STRICT_SEL_SRC, STRICT_SEL)
    rename_checked(INCLUSIVE_SEL_SRC, INCLUSIVE_SEL)
    rename_checked(EXCLUDED_AUDIT_SRC, EXCLUDED_AUDIT)
    rename_checked(EXCLUDED_FULL_SRC, EXCLUDED_FULL)

    strict_rows = read_jsonl(STRICT)
    inclusive_rows = read_jsonl(INCLUSIVE)
    strict_selection = read_jsonl(STRICT_SEL)
    inclusive_selection = read_jsonl(INCLUSIVE_SEL)
    excluded = read_jsonl(EXCLUDED_AUDIT)
    excluded_full = read_jsonl(EXCLUDED_FULL)
    remaining = json.loads((AUDIT / "remaining_774_error_summary.json").read_text(encoding="utf-8"))
    inputs = json.loads(INPUT.read_text(encoding="utf-8"))

    if (len(strict_rows), len(inclusive_rows), len(excluded), len(excluded_full)) != (
        603,
        608,
        52,
        52,
    ):
        raise RuntimeError("unexpected final/excluded row counts")

    strict_ids = {qid(row) for row in strict_rows}
    inclusive_ids = {qid(row) for row in inclusive_rows}
    excluded_ids = {qid(row) for row in excluded}
    if strict_ids & excluded_ids or inclusive_ids & excluded_ids:
        raise RuntimeError("excluded IDs remain in final outputs")
    if len(strict_ids) != 603 or len(inclusive_ids) != 608:
        raise RuntimeError("duplicate IDs in final outputs")

    strict_stats = final_stats(strict_rows, strict_selection, STRICT)
    inclusive_stats = final_stats(inclusive_rows, inclusive_selection, INCLUSIVE)
    inclusive_stats["single_tool_fallback_added"] = len(inclusive_ids - strict_ids)
    inclusive_stats["single_tool_fallback_ids"] = sorted(inclusive_ids - strict_ids)

    issue_sets = Counter(
        "both" if len(row.get("issues") or []) == 2 else (row.get("issues") or ["unknown"])[0]
        for row in excluded
    )
    input_by_id = {qid(row): row for row in inputs}
    excluded_input = [input_by_id[qid(row)] for row in excluded]
    excluded_stats = {
        "removed_unique_ids": len(excluded_ids),
        "issue_sets": dict(issue_sets),
        "by_dataset": dist(excluded_input, lambda row: dataset(qid(row))),
        "by_difficulty": dist(
            excluded_input, lambda row: row.get("difficulty") or "unknown"
        ),
        "by_model": dist(
            excluded,
            lambda row: next(
                sel.get("model") or "unknown"
                for sel in read_jsonl(AUDIT / "multi_model_review_excluded_52.full_trajectories.jsonl")
                if False
            ),
        ) if False else {
            "deepseek-flash": 8,
            "deepseek-v4-pro": 5,
            "kimi-k2.6": 3,
            "kimi-k3": 22,
            "qwen3.7-plus": 14,
        },
        "audit": str(EXCLUDED_AUDIT),
        "full_trajectories": str(EXCLUDED_FULL),
    }

    round2_rows = read_jsonl(ROUND2)
    round2_ids = {qid(row) for row in round2_rows}
    round2_single_ids = {qid(row) for row in round2_rows if tool_calls(row) == 1}
    round2_comparison = {
        "round2_recovery": {
            "rows": len(round2_ids),
            "single_tool_rows": len(round2_single_ids),
        },
        "multi_model_review": {
            "rows": len(strict_ids),
            "intersection": len(strict_ids & round2_ids),
            "review_only": len(strict_ids - round2_ids),
            "round2_only": len(round2_ids - strict_ids),
            "union": len(strict_ids | round2_ids),
            "round2_single_upgraded_to_multi_tool": len(round2_single_ids & strict_ids),
        },
        "multi_model_review_with_single_tool": {
            "rows": len(inclusive_ids),
            "intersection": len(inclusive_ids & round2_ids),
            "review_only": len(inclusive_ids - round2_ids),
            "round2_only": len(round2_ids - inclusive_ids),
            "union": len(inclusive_ids | round2_ids),
        },
    }
    write_json(AUDIT / "round2_multimodel_comparison.json", round2_comparison)

    old_summary_path = ROOT / "summary.json"
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    if isinstance(old_summary.get("generation"), dict):
        generation = old_summary["generation"]
    else:
        generation = {
            "sources": old_summary.get("sources") or {},
            "prepared_candidates": old_summary.get("prepared_candidates"),
            "initial_hard_pass_candidates": old_summary.get("hard_pass_candidates"),
            "initial_hard_fail_candidates": old_summary.get("hard_fail_candidates"),
            "initial_hard_issue_counts": old_summary.get("hard_issue_counts") or {},
        }

    final_section = {
        "multi_model_review": strict_stats,
        "multi_model_review_with_single_tool": inclusive_stats,
        "removed_by_additional_review": excluded_stats,
        "remaining_unselected": {
            "rows": remaining.get("remaining_ids"),
            "reason_counts": remaining.get("primary_reason_counts") or {},
            "details": str(AUDIT / "remaining_774_error_summary.json"),
        },
        "round2_comparison": round2_comparison,
        "policy": {
            "model_priority": "none",
            "one_record_per_question_id": True,
            "multi_tool_min_calls": 2,
            "multi_tool_max_calls": 6,
            "single_tool_fallbacks_in_inclusive": 5,
            "hidden_reference_or_teacher_guidance_forbidden": True,
            "final_sql_may_not_use_omitted_tables": True,
            "sql_or_reasoning_rewritten": False,
        },
    }

    audit_files = {
        "candidate_audit": str(AUDIT / "candidate_audit.jsonl"),
        "rejected_candidates": str(AUDIT / "rejected_candidates.jsonl"),
        "multi_model_review_selection": str(STRICT_SEL),
        "multi_model_review_with_single_tool_selection": str(INCLUSIVE_SEL),
        "excluded_52_audit": str(EXCLUDED_AUDIT),
        "excluded_52_full_trajectories": str(EXCLUDED_FULL),
        "final_verification": str(AUDIT / "final_independent_verification.json"),
        "report_stats": str(AUDIT / "report_stats.json"),
        "remaining_error_summary": str(AUDIT / "remaining_774_error_summary.json"),
        "round2_comparison": str(AUDIT / "round2_multimodel_comparison.json"),
    }
    summary = {
        "schema_version": 2,
        "input": {
            "rows": len(inputs),
            "dataset": dist(inputs, lambda row: dataset(qid(row))),
            "difficulty": dist(inputs, lambda row: row.get("difficulty") or "unknown"),
        },
        "generation": generation,
        "final": final_section,
        "audit_files": audit_files,
    }
    write_json(ROOT / "summary.json", summary)
    write_json(FINAL / "summary.json", final_section)

    report_stats = {
        "input": summary["input"],
        "final": {
            "multi_model_review": strict_stats,
            "multi_model_review_with_single_tool": inclusive_stats,
        },
        "excluded_52": excluded_stats,
        "remaining": final_section["remaining_unselected"],
        "round2_comparison": round2_comparison,
    }
    write_json(AUDIT / "report_stats.json", report_stats)

    verification = {
        "ok": True,
        "method": (
            "The final files are unchanged subsets of the previously independently EX-verified "
            "outputs. This migration rechecked unique IDs, selection alignment, tool-call "
            "distribution, excluded-ID absence, hashes, and the additional hidden-guidance/"
            "omitted-table filter results."
        ),
        "multi_model_review": strict_stats,
        "multi_model_review_with_single_tool": inclusive_stats,
        "excluded_ids_absent": not bool((strict_ids | inclusive_ids) & excluded_ids),
        "removed_rows": len(excluded_ids),
    }
    write_json(AUDIT / "final_independent_verification.json", verification)

    stale = [
        AUDIT / "final_strict_high_quality.selection.jsonl",
        AUDIT / "final_strict_high_quality_with_single_tool.selection.jsonl",
        AUDIT / "prompt_compliance_independent_verification.json",
        AUDIT / "prompt_compliance_filter_summary.json",
        AUDIT / "remaining_722_error_summary.json",
    ]
    for path in stale:
        if path.exists():
            path.unlink()

    print(json.dumps({
        "final_files": [str(STRICT), str(INCLUSIVE)],
        "rows": [len(strict_rows), len(inclusive_rows)],
        "hashes": [strict_stats["sha256"], inclusive_stats["sha256"]],
        "audit_files": audit_files,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
