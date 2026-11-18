#!/usr/bin/env python3
"""Materialize and merge greedy generation-stage pass/fail records."""

from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path("/data/keenSQL")
SCRIPT_ROOT = ROOT / "train_prompt/sft_easy_teacher_traces"
DATA_ROOT = ROOT / "dataset/sft_data/easy_teacher_traces"
INPUT_ROOT = DATA_ROOT / "inputs"
MODEL_ROOT = DATA_ROOT / "models/qwen3.6-plus/greedy"
OUTPUT_ROOT = DATA_ROOT / "merged/generation_stage/qwen3.6-plus/greedy"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object at {path}:{line_number}")
            yield row


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_builder():
    path = SCRIPT_ROOT / "build_easy_teacher_traces.py"
    spec = importlib.util.spec_from_file_location("_easy_builder_for_merge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    builder = load_builder()
    global_rows = json.loads((INPUT_ROOT / "easy_current.json").read_text(encoding="utf-8"))
    order = {str(row["question_id"]): index for index, row in enumerate(global_rows)}
    passed_by_qid: dict[str, dict[str, Any]] = {}
    failed_by_qid: dict[str, dict[str, Any]] = {}
    issues: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    terminal_warnings: Counter[str] = Counter()
    reclassified: Counter[str] = Counter()

    for chunk_path in sorted((INPUT_ROOT / "chunks").glob("chunk_*.json")):
        chunk = chunk_path.stem
        chunk_rows = json.loads(chunk_path.read_text(encoding="utf-8"))
        output_dir = MODEL_ROOT / chunk
        raw_path = output_dir / "raw_attempts.jsonl"
        latest: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(raw_path):
            latest[str(row["attempt_id"])] = row
        if len(latest) != len(chunk_rows):
            raise ValueError(
                f"{chunk}: attempts={len(latest)} finalized_inputs={len(chunk_rows)}"
            )

        # Refresh stale per-chunk accepted/rejected/summary files from raw data.
        builder.adapter.core.materialize_run(output_dir)

        for row in latest.values():
            qid = str(row.get("question_id") or "")
            if qid not in order:
                raise ValueError(f"raw attempt outside finalized input: {qid}")
            acceptance = row.get("acceptance") or {}
            issues.update(acceptance.get("issues") or [])
            warnings.update(acceptance.get("teacher_warnings") or [])
            episode = row.get("episode") or {}
            grade = row.get("grade") or {}
            hard_failures: list[str] = []
            if row.get("accepted") is True:
                if acceptance.get("issues"):
                    hard_failures.append("accepted_with_generation_issues")
                if grade.get("ex") is not True:
                    hard_failures.append("accepted_without_ex_true")
                if not isinstance(row.get("sft_record"), dict):
                    hard_failures.append("accepted_without_sft_record")
                if episode.get("final_was_executed") is not True:
                    hard_failures.append("accepted_without_final_execution")
                if (episode.get("verification") or {}).get("matches") is not True:
                    hard_failures.append("accepted_without_verified_match")
                tool_calls = int(episode.get("tool_calls") or 0)
                if not 2 <= tool_calls <= 6:
                    hard_failures.append("accepted_tool_calls_outside_2_to_6")

                calls = episode.get("executed_calls") or []
                final_sql = str(episode.get("final_sql") or "").strip()
                if calls and str(calls[-1].get("sql") or "").strip() != final_sql:
                    terminal_warnings["final_sql_not_last_tool_call"] += 1
                declared = (episode.get("verification") or {}).get("declared")
                if calls and declared != calls[-1].get("index"):
                    terminal_warnings["verified_call_is_not_last_tool_call"] += 1

            if row.get("accepted") is True and not hard_failures:
                if qid in passed_by_qid or qid in failed_by_qid:
                    raise ValueError(f"duplicate merged question_id: {qid}")
                passed_by_qid[qid] = row["sft_record"]
            else:
                if qid in passed_by_qid or qid in failed_by_qid:
                    raise ValueError(f"duplicate merged question_id: {qid}")
                rejected = {key: value for key, value in row.items() if key != "sft_record"}
                if hard_failures:
                    rejected["merge_reclassification_issues"] = hard_failures
                    reclassified.update(hard_failures)
                failed_by_qid[qid] = rejected

    if set(passed_by_qid) & set(failed_by_qid):
        raise ValueError("pass/fail question IDs overlap")
    if set(passed_by_qid) | set(failed_by_qid) != set(order):
        raise ValueError("pass/fail union does not equal finalized input")

    passed = sorted(passed_by_qid.values(), key=lambda row: order[str(row["question_id"])])
    failed = sorted(failed_by_qid.values(), key=lambda row: order[str(row["question_id"])])
    pass_path = OUTPUT_ROOT / "generation_stage_strict_pass.jsonl"
    fail_path = OUTPUT_ROOT / "generation_stage_rejected.jsonl"
    pass_count = atomic_jsonl(pass_path, passed)
    fail_count = atomic_jsonl(fail_path, failed)
    summary = {
        "classification": (
            "generation-stage accepted flag plus generation hard invariants; "
            "no independent replay audit"
        ),
        "finalized_input_records": len(order),
        "strict_pass_records": pass_count,
        "rejected_records": fail_count,
        "unique_question_ids": len(set(order)),
        "pass_fail_overlap": 0,
        "generation_issue_counts": dict(issues),
        "generation_warning_counts": dict(warnings),
        "terminal_call_warnings": dict(terminal_warnings),
        "merge_reclassification_counts": dict(reclassified),
        "strict_pass_output": str(pass_path),
        "rejected_output": str(fail_path),
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
