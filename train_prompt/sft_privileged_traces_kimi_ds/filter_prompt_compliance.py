#!/usr/bin/env python3
"""Filter hidden-teacher phrasing and shown-schema violations from final SFT files."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from sqlglot import exp, parse_one


ROOT = Path(
    "/data/keenSQL/dataset/sft_data/"
    "privileged_traces_sft_privileged_traces_kimi_ds"
)
FINAL = ROOT / "all_output_merge/final"
AUDIT = ROOT / "all_output_merge/audit"
STRICT_IN = FINAL / "strict_high_quality.jsonl"
INCLUSIVE_IN = FINAL / "strict_high_quality_with_single_tool.jsonl"
STRICT_OUT = FINAL / "strict_high_quality_prompt_compliant.jsonl"
INCLUSIVE_OUT = FINAL / "strict_high_quality_with_single_tool_prompt_compliant.jsonl"
EXCLUDED_OUT = AUDIT / "prompt_compliance_excluded_52.jsonl"
EXCLUDED_FULL_OUT = AUDIT / "prompt_compliance_excluded_52_full_trajectories.jsonl"
SUMMARY_OUT = AUDIT / "prompt_compliance_filter_summary.json"
VERIFY_OUT = AUDIT / "prompt_compliance_independent_verification.json"
OLD_REMAINING = AUDIT / "remaining_722_error_summary.json"
NEW_REMAINING = AUDIT / "remaining_774_error_summary.json"

STRICT_SELECTION_IN = AUDIT / "final_strict_high_quality.selection.jsonl"
INCLUSIVE_SELECTION_IN = AUDIT / "final_strict_high_quality_with_single_tool.selection.jsonl"
STRICT_SELECTION_OUT = AUDIT / "final_strict_high_quality_prompt_compliant.selection.jsonl"
INCLUSIVE_SELECTION_OUT = (
    AUDIT / "final_strict_high_quality_with_single_tool_prompt_compliant.selection.jsonl"
)

INPUT = Path(
    "/data/keenSQL/dataset/sft_data/privileged_traces/"
    "other_teacher_input/remaining_hard_1382.json"
)
ROUND2 = Path(
    "/data/keenSQL/dataset/sft_data/privileged_traces/round2_recovery/"
    "merged/sft_privileged_round2_merged.jsonl"
)

HIDDEN_GUIDANCE = re.compile(
    r"(?i)(?:"
    r"matching\s+reference|"
    r"follow\s+(?:the\s+)?reference|"
    r"per\s+reference|"
    r"reference(?:-shaped|-style)?\s+"
    r"(?:semantics|reading|framing|logic|pattern|convention|uses|projects|join)|"
    r"teacher\s+(?:semantics|semantic\s+guidance|guidance|formula)"
    r")"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def qid(row: dict) -> str:
    return str(row.get("question_id") or "")


def assistant_text(row: dict) -> str:
    start = int(row.get("prompt_messages") or 0)
    return "\n".join(
        str(item.get("content") or "")
        for item in row.get("messages") or []
        if isinstance(item, dict)
        and item.get("role") == "assistant"
        and (row.get("messages") or []).index(item) >= start
    )


def post_prompt_assistant_text(row: dict) -> str:
    start = int(row.get("prompt_messages") or 0)
    return "\n".join(
        str(item.get("content") or "")
        for item in (row.get("messages") or [])[start:]
        if isinstance(item, dict) and item.get("role") == "assistant"
    )


def final_sql(text: str) -> str:
    matches = re.findall(r"(?is)<final_answer>\s*(.*?)\s*</final_answer>", text)
    if len(matches) != 1:
        raise ValueError(f"expected one final_answer, got {len(matches)}")
    return matches[0].strip()


def omitted_tables(row: dict) -> set[str]:
    start = int(row.get("prompt_messages") or 0)
    if start < 1:
        return set()
    prompt = str((row.get("messages") or [])[start - 1].get("content") or "")
    values: set[str] = set()
    for match in re.finditer(r"(?im)^# Omitted tables:\s*(.+)$", prompt):
        values.update(
            value.strip().lower()
            for value in match.group(1).split(",")
            if value.strip()
        )
    return values


def sql_tables(sql: str) -> set[str]:
    tree = parse_one(sql, read="sqlite")
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    return {table.name.lower() for table in tree.find_all(exp.Table)} - ctes


def audit_row(row: dict) -> dict:
    text = post_prompt_assistant_text(row)
    sql = final_sql(text)
    hidden_matches = sorted({match.group(0) for match in HIDDEN_GUIDANCE.finditer(text)})
    omitted = omitted_tables(row)
    tables = sql_tables(sql)
    omitted_used = sorted(omitted & tables)
    issues: list[str] = []
    if hidden_matches:
        issues.append("hidden_reference_or_teacher_guidance")
    if omitted_used:
        issues.append("final_sql_uses_omitted_table")
    return {
        "question_id": qid(row),
        "record_id": row.get("id"),
        "issues": issues,
        "hidden_guidance_matches": hidden_matches,
        "omitted_tables_used": omitted_used,
        "final_sql": sql,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset(question_id: str) -> str:
    if question_id.startswith("bird_"):
        return "bird_train"
    if question_id.startswith("omni_"):
        return "omni-2k"
    if question_id.startswith("spider_"):
        return "spider1"
    return "bird-verified-2.5k"


def tool_calls(row: dict) -> int:
    return len(re.findall(r"(?i)<tool_call>", post_prompt_assistant_text(row)))


def dist(rows: list[dict], key) -> dict[str, int]:
    return dict(sorted(Counter(key(row) for row in rows).items()))


def main() -> None:
    strict = load_jsonl(STRICT_IN)
    inclusive = load_jsonl(INCLUSIVE_IN)
    strict_selection = load_jsonl(STRICT_SELECTION_IN)
    inclusive_selection = load_jsonl(INCLUSIVE_SELECTION_IN)
    selection_by_id = {qid(row): row for row in inclusive_selection}
    input_by_id = {qid(row): row for row in json.loads(INPUT.read_text(encoding="utf-8"))}

    audited = [(row, audit_row(row)) for row in strict]
    excluded = [audit for _row, audit in audited if audit["issues"]]
    excluded_ids = {audit["question_id"] for audit in excluded}
    strict_clean = [row for row, audit in audited if not audit["issues"]]

    # Apply the same two checks to the inclusive file, including its five one-tool additions.
    inclusive_audited = [(row, audit_row(row)) for row in inclusive]
    inclusive_clean = [row for row, audit in inclusive_audited if not audit["issues"]]
    inclusive_extra = [row for row in inclusive_clean if qid(row) not in {qid(x) for x in strict_clean}]

    write_jsonl(STRICT_OUT, strict_clean)
    write_jsonl(INCLUSIVE_OUT, inclusive_clean)
    write_jsonl(EXCLUDED_OUT, excluded)
    write_jsonl(
        EXCLUDED_FULL_OUT,
        [row for row, audit in audited if audit["issues"]],
    )
    write_jsonl(
        STRICT_SELECTION_OUT,
        [row for row in strict_selection if qid(row) not in excluded_ids],
    )
    write_jsonl(
        INCLUSIVE_SELECTION_OUT,
        [row for row in inclusive_selection if qid(row) not in excluded_ids],
    )

    issue_sets = Counter(
        "both"
        if len(audit["issues"]) == 2
        else audit["issues"][0]
        for audit in excluded
    )
    removed_rows = [input_by_id[audit["question_id"]] for audit in excluded]

    strict_sel_clean = [row for row in strict_selection if qid(row) not in excluded_ids]
    inclusive_sel_clean = [row for row in inclusive_selection if qid(row) not in excluded_ids]

    round2_rows = load_jsonl(ROUND2)
    round2_ids = {qid(row) for row in round2_rows}
    round2_single_ids = {
        qid(row)
        for row in round2_rows
        if len(re.findall(r"(?i)<tool_call>", post_prompt_assistant_text(row))) == 1
    }
    strict_ids = {qid(row) for row in strict_clean}
    inclusive_ids = {qid(row) for row in inclusive_clean}

    example_ids = {"4333", "omni_1273", "bird_01005"}

    previous_remaining = json.loads(OLD_REMAINING.read_text(encoding="utf-8"))
    remaining_details = list(previous_remaining.get("details") or [])
    for audit in excluded:
        question_id = audit["question_id"]
        meta = input_by_id[question_id]
        if len(audit["issues"]) == 2:
            primary_reason = "hidden_guidance_and_omitted_table"
        elif audit["issues"][0] == "hidden_reference_or_teacher_guidance":
            primary_reason = "hidden_reference_or_teacher_guidance"
        else:
            primary_reason = "final_sql_uses_omitted_table"
        remaining_details.append(
            {
                "question_id": question_id,
                "db_id": meta.get("db_id"),
                "dataset": dataset(question_id),
                "difficulty": meta.get("difficulty") or "unknown",
                "primary_reason": primary_reason,
                "prompt_compliance_audit": audit,
            }
        )
    remaining_details.sort(key=lambda row: row["question_id"])
    remaining_counts = Counter(row["primary_reason"] for row in remaining_details)
    remaining_by_difficulty: dict[str, Counter[str]] = {}
    remaining_by_dataset: dict[str, Counter[str]] = {}
    for row in remaining_details:
        remaining_by_difficulty.setdefault(row["difficulty"], Counter())[row["primary_reason"]] += 1
        remaining_by_dataset.setdefault(row["dataset"], Counter())[row["primary_reason"]] += 1
    remaining_summary = {
        "input_ids": len(input_by_id),
        "selected_ids": len(inclusive_clean),
        "remaining_ids": len(remaining_details),
        "definition": "Each unselected ID has one mutually exclusive primary reason after the prompt-compliance filter.",
        "primary_reason_counts": dict(remaining_counts.most_common()),
        "by_difficulty": {
            key: dict(value.most_common())
            for key, value in sorted(remaining_by_difficulty.items())
        },
        "by_dataset": {
            key: dict(value.most_common())
            for key, value in sorted(remaining_by_dataset.items())
        },
        "details": remaining_details,
    }
    NEW_REMAINING.write_text(
        json.dumps(remaining_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "source": {
            "strict_rows": len(strict),
            "inclusive_rows": len(inclusive),
        },
        "excluded": {
            "rows": len(excluded),
            "issue_sets": dict(issue_sets),
            "dataset": dist(removed_rows, lambda row: dataset(qid(row))),
            "difficulty": dist(removed_rows, lambda row: str(row.get("difficulty") or "unknown")),
            "model": dist(excluded, lambda row: str(selection_by_id[qid(row)].get("model") or "unknown")),
            "tool_calls": dist(
                [row for row, audit in audited if audit["issues"]], tool_calls
            ),
        },
        "strict_prompt_compliant": {
            "rows": len(strict_clean),
            "sha256": sha256(STRICT_OUT),
            "dataset": dist(strict_clean, lambda row: dataset(qid(row))),
            "difficulty": dist(
                strict_sel_clean, lambda row: str(row.get("difficulty") or "unknown")
            ),
            "model": dist(strict_sel_clean, lambda row: str(row.get("model") or "unknown")),
            "tool_calls": dist(strict_clean, tool_calls),
        },
        "inclusive_prompt_compliant": {
            "rows": len(inclusive_clean),
            "sha256": sha256(INCLUSIVE_OUT),
            "dataset": dist(inclusive_clean, lambda row: dataset(qid(row))),
            "difficulty": dist(
                inclusive_sel_clean, lambda row: str(row.get("difficulty") or "unknown")
            ),
            "model": dist(inclusive_sel_clean, lambda row: str(row.get("model") or "unknown")),
            "tool_calls": dist(inclusive_clean, tool_calls),
            "single_tool_added": len(inclusive_extra),
            "single_tool_ids": [qid(row) for row in inclusive_extra],
        },
        "round2_comparison": {
            "round2_rows": len(round2_ids),
            "round2_single_tool_rows": len(round2_single_ids),
            "round2_single_upgraded_to_clean_multi_tool": len(round2_single_ids & strict_ids),
            "strict_intersection": len(round2_ids & strict_ids),
            "strict_union": len(round2_ids | strict_ids),
            "inclusive_intersection": len(round2_ids & inclusive_ids),
            "inclusive_union": len(round2_ids | inclusive_ids),
        },
        "report_examples": {
            "expected": sorted(example_ids),
            "excluded": sorted(example_ids & excluded_ids),
            "retained": sorted(example_ids & strict_ids),
        },
        "paths": {
            "strict": str(STRICT_OUT),
            "inclusive": str(INCLUSIVE_OUT),
            "excluded": str(EXCLUDED_OUT),
            "excluded_full_trajectories": str(EXCLUDED_FULL_OUT),
            "strict_selection": str(STRICT_SELECTION_OUT),
            "inclusive_selection": str(INCLUSIVE_SELECTION_OUT),
            "remaining_error_summary": str(NEW_REMAINING),
            "verification": str(VERIFY_OUT),
        },
    }
    SUMMARY_OUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verification = {
        "ok": True,
        "method": (
            "The new files are unchanged subsets of the previously independently EX-verified "
            "files, followed by a fresh hidden-guidance and omitted-table audit."
        ),
        "strict": {
            "rows": len(strict_clean),
            "unique_ids": len({qid(row) for row in strict_clean}),
            "subset_of_original": {qid(row) for row in strict_clean}
            <= {qid(row) for row in strict},
            "prompt_compliance_violations": sum(
                bool(audit_row(row)["issues"]) for row in strict_clean
            ),
            "tool_calls": dist(strict_clean, tool_calls),
            "sha256": sha256(STRICT_OUT),
        },
        "inclusive": {
            "rows": len(inclusive_clean),
            "unique_ids": len({qid(row) for row in inclusive_clean}),
            "subset_of_original": {qid(row) for row in inclusive_clean}
            <= {qid(row) for row in inclusive},
            "prompt_compliance_violations": sum(
                bool(audit_row(row)["issues"]) for row in inclusive_clean
            ),
            "tool_calls": dist(inclusive_clean, tool_calls),
            "sha256": sha256(INCLUSIVE_OUT),
        },
        "source_independent_ex_verification": str(
            AUDIT / "final_independent_verification.json"
        ),
    }
    VERIFY_OUT.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
