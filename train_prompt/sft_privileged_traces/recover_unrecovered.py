#!/usr/bin/env python3
"""Recover hard-band samples that were rejected for non-paper reasons.

The two generation rounds produced 1,440 accepted privileged traces out of
2,462 hard questions. The remaining 1,022 classify as:

``A`` 559  no EX-correct attempt in any of the 36 decodings -- genuinely unsolved
``B`` 240  EX-correct, but the exact final SQL never returned a successful
          observation in-trace (budget/step-limit or never executed)
``C`` 160  EX-correct with a successful terminal execution, rejected only by the
          trajectory protocol or a leak rule
``D``  63  already accepted, then removed by a downstream literal filter

Rejecting B/C on the final-observation rule is stricter than the paper, which
verifies traces by executing the final SQL offline and checking EX. This script
recovers only the subsets where the decision is mechanical, and tags every
record with its tier so an ablation can drop any tier.

Tiers:

``B_budget``
    Round-2 rows with ``acceptance.issues == []`` whose only defect is
    ``final_observation_error:step_limit_exceeded``. The final SQL is re-executed
    here and must still match gold.
``C_pointer``
    Every protocol issue is a ``Verified call`` pointer code, and one
    ``sqlite_query`` provably executed the exact final SQL successfully.
``C_single_call``
    Rejected only by the first-round ``tool_calls_below_required_2`` rule. The
    round-2 protocol accepts one call, and the final SQL was executed exactly.
``D_literal_review``
    Removed by the temporal gold-literal filter. Re-admitted only when every
    flagged literal is a date/number scaffold rather than an answer entity.

Every accepted row is re-graded for EX and re-checked with the shared protocol
validator, so a tier tag can never substitute for correctness.

Usage:
    python recover_unrecovered.py recover
    python recover_unrecovered.py clean-base
    python recover_unrecovered.py merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path("/data/keenSQL")
TRACE_DIR = PROJECT_ROOT / "train_prompt/sft_privileged_traces"
for path in (
    TRACE_DIR,
    PROJECT_ROOT,
    PROJECT_ROOT / "train_prompt",
    PROJECT_ROOT / "train_prompt/retoolsql_prompt",
    PROJECT_ROOT / "database_tools",
):
    sys.path.insert(0, str(path))

import prompt_builder as PB  # noqa: E402
import build_sft_traces as B  # noqa: E402
from common.protocol import protocol_errors, split_protocol_errors  # noqa: E402
from common.trace_repair import executed_final_call, repair_verified_call  # noqa: E402

SFT_DATA = PROJECT_ROOT / "dataset/sft_data"
PRIV = SFT_DATA / "privileged_traces"
HARD_POOL = SFT_DATA / "data/all_dataset_qwen35_9b_pass16_wrong.json"
FINAL_2 = SFT_DATA / "data/sft_privileged_2.jsonl"
ROUND2 = PRIV / "round2_recovery"
BACKUP = ROUND2 / "backup_before_literal_filter_20261024"
SCHEMA_CACHE = PROJECT_ROOT / "dataset/m_schema/cache"

BRANCHES = ("qwen35-greedy", "qwen35-pass8", "qwen36plus-greedy", "qwen36plus-pass8")
POINTER_CODES = {
    "missing_verified_call",
    "verified_call_out_of_range",
    "verified_call_sql_mismatch",
}
SINGLE_CALL_CODE = "tool_calls_below_required_2"
ATTEMPT_ALLOWED_CODES = POINTER_CODES | {SINGLE_CALL_CODE}
META_REFERENCE_MARKERS = ("correct SQL", "Oracle SQL", "teacher note")

# Date/format directives and pure numbers carry no answer semantics on their
# own; a gold literal drawn only from these is a formatting artifact.
BENIGN_LITERAL = re.compile(
    r"^(?:%[A-Za-z%]|[\d\s:/.\-+%,]+|(?:19|20)\d{2}(?:-\d{2}-\d{2})?)$"
)

# Gold result sets keyed by (db file, file stamp, gold SQL). Shared across a
# run so a sample's gold query executes once, not once per attempt.
_GOLD_CACHE: Dict[Tuple[str, Tuple[int, int], str], Any] = {}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_hard_pool() -> Dict[int, Dict[str, Any]]:
    return {
        index: row
        for index, row in enumerate(json.loads(HARD_POOL.read_text(encoding="utf-8")))
    }


def accepted_source_indexes() -> set[int]:
    return {
        int(row["metadata"]["source_index"])
        for row in iter_jsonl(FINAL_2)
        if (row.get("metadata") or {}).get("source_index") is not None
    }


def unrecovered_indexes() -> set[int]:
    return set(load_hard_pool()) - accepted_source_indexes()


def rebuild_deployment_prefix(
    hard: Mapping[str, Any],
    prompt_meta: Mapping[str, Any],
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """Rebuild the exact student-visible prefix a teacher run used.

    The raw attempt rows store only the transcript, so the prefix is rebuilt
    from the same schema cache and the same base budget recorded at generation
    time. The rebuilt tier and token count must match the stored metadata;
    otherwise the prefix is a different context and the row is skipped.
    """

    db_id = str(hard.get("db_id") or "")
    schema_path = SCHEMA_CACHE / f"{db_id}.txt"
    if not schema_path.is_file():
        return None, "missing_schema"
    budget = int(prompt_meta.get("base_prompt_budget") or 0)
    if budget <= 0:
        return None, "missing_base_prompt_budget"
    try:
        built = PB.build_messages(
            engine="SQLite",
            db_details=schema_path.read_text(encoding="utf-8"),
            external_knowledge=str(hard.get("evidence") or ""),
            question=str(hard.get("question") or ""),
            fewshot=bool(prompt_meta.get("fewshot", True)),
            max_prompt_tokens=budget,
        )
    except PB.SchemaBudgetError:
        return None, "schema_budget"
    meta = built["meta"]
    if int(meta.get("schema_tier", -1)) != int(prompt_meta.get("schema_tier", -2)):
        return None, "schema_tier_mismatch"
    if abs(int(meta.get("prompt_tokens", 0)) - int(prompt_meta.get("prompt_tokens", -1))) > 2:
        return None, "prompt_tokens_mismatch"
    return built["messages"], None


def append_transcript(
    prefix: Sequence[Mapping[str, Any]], transcript: Sequence[Mapping[str, Any]]
) -> Optional[List[Dict[str, str]]]:
    messages = [{"role": str(item["role"]), "content": str(item.get("content") or "")} for item in prefix]
    for item in transcript:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role == "assistant":
            messages.append({"role": "assistant", "content": content})
        elif role in {"observation", "tool"}:
            messages.append({"role": "user", "content": f"<observation>\n{content}\n</observation>"})
        else:
            return None
    return messages


def final_sql_of(messages: Sequence[Mapping[str, Any]]) -> Optional[str]:
    parsed = B.T.parse_assistant_turn(
        str(messages[-1].get("content") or ""),
        require_final_scratch_pad=False,
        require_tool_scratch_pad=False,
    )
    return parsed.get("sql")


def regrade(
    messages: Sequence[Mapping[str, Any]],
    hard: Mapping[str, Any],
    *,
    db_root: Path,
) -> Dict[str, Any]:
    """Grade the final SQL against gold, caching both executions.

    ``B.grade_final`` re-runs the gold statement on every call. A sample's gold
    query is constant across its attempts, and some of them are expensive, so
    the recovery pass would otherwise spend most of its time repeating work.
    The comparison still goes through the same ``grade_results`` used by the
    screening harness.
    """

    db_file = B.db_file_for(db_root, str(hard.get("db_id") or ""))
    if db_file is None:
        return {"ex": None, "reason": "database_missing"}
    final_sql = str(final_sql_of(messages) or "")
    method = str(hard.get("grading_method") or "multiset")
    gold_sql = str(hard.get("SQL") or "")

    gold_key = (str(db_file), _file_stamp(db_file), gold_sql)
    try:
        gold_rows = _GOLD_CACHE[gold_key]
    except KeyError:
        try:
            gold_rows = B.execute_query(db_file, gold_sql)
        except Exception as exc:  # noqa: BLE001 - reported like grade_final does
            _GOLD_CACHE[gold_key] = exc
            return {
                "ex": None,
                "reason": f"gold execution failed: {type(exc).__name__}: {exc}",
                "gold_row_count": None,
                "pred_row_count": None,
            }
        _GOLD_CACHE[gold_key] = gold_rows
    if isinstance(gold_rows, Exception):
        return {
            "ex": None,
            "reason": f"gold execution failed: {type(gold_rows).__name__}: {gold_rows}",
            "gold_row_count": None,
            "pred_row_count": None,
        }

    try:
        predicted_rows = B.execute_query(db_file, final_sql)
    except Exception as exc:  # noqa: BLE001
        return {
            "ex": False,
            "reason": f"prediction execution failed: {type(exc).__name__}: {exc}",
            "gold_row_count": len(gold_rows),
            "pred_row_count": None,
        }
    try:
        matched = bool(B.grade_results(gold_rows, predicted_rows, method))
    except Exception as exc:  # noqa: BLE001
        return {
            "ex": None,
            "reason": f"grading failed: {type(exc).__name__}: {exc}",
            "gold_row_count": len(gold_rows),
            "pred_row_count": len(predicted_rows),
        }
    return {
        "ex": matched,
        "reason": "match" if matched else "mismatch",
        "gold_row_count": len(gold_rows),
        "pred_row_count": len(predicted_rows),
    }


def _file_stamp(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def has_meta_reference(messages: Sequence[Mapping[str, Any]], prompt_messages: int) -> bool:
    text = "\n".join(
        str(item.get("content") or "")
        for item in messages[prompt_messages:]
        if item.get("role") == "assistant"
    )
    return any(marker in text for marker in META_REFERENCE_MARKERS)


def base_record(
    *,
    hard: Mapping[str, Any],
    messages: List[Dict[str, str]],
    prompt_messages: int,
    prompt_meta: Mapping[str, Any],
    model: str,
    decoding: str,
    attempt: int,
    source_index: int,
    tier: str,
) -> Dict[str, Any]:
    trajectory_hash = B.sha256_text(
        json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    sample_id = B.sample_id(source_index, hard)
    return {
        "id": f"{sample_id}-a{attempt}-{trajectory_hash[:12]}",
        "messages": messages,
        "prompt_messages": prompt_messages,
        "prompt_hash": prompt_meta.get("prompt_sha256"),
        "prompt_version": prompt_meta.get("prompt_version"),
        "teacher_prompt_sha256": prompt_meta.get("teacher_prompt_sha256"),
        "question_id": hard.get("question_id"),
        "question": hard.get("question"),
        "evidence": hard.get("evidence"),
        "db_id": hard.get("db_id"),
        "gold_sql": str(hard.get("SQL") or "").strip(),
        "grading_method": str(hard.get("grading_method") or "multiset").strip(),
        "data_variant": "hard_privileged",
        "metadata": {
            "sample_id": sample_id,
            "source_index": source_index,
            "question_id": hard.get("question_id"),
            "db_id": hard.get("db_id"),
            "model": model,
            "decoding": decoding,
            "attempt": attempt,
            "grading_method": str(hard.get("grading_method") or "multiset").strip(),
            "ex": True,
            "tool_calls": prompt_meta.get("tool_calls"),
            "schema_tier": prompt_meta.get("schema_tier"),
            "teacher_prompt_tokens": prompt_meta.get("teacher_prompt_tokens"),
            "gold_sql_sha256": B.sha256_text(str(hard.get("SQL") or "").strip()),
            "teacher_context_in_messages": False,
            "loss_roles": ["assistant"],
            "recovery_tier": tier,
        },
    }


def recover_budget(*, out_dir: Path, db_root: Path) -> Dict[str, Any]:
    """Tier B: round-2 traces whose exact final call hit the step limit."""

    unrecovered = unrecovered_indexes()
    by_index = load_hard_pool()
    best: Dict[int, Dict[str, Any]] = {}
    rejected: Counter = Counter()
    candidates = 0

    for branch in BRANCHES:
        audit_path = ROUND2 / branch / "accepted_sft.final_success.rejected.jsonl"
        source_path = ROUND2 / branch / "accepted_sft.jsonl"
        if not audit_path.is_file() or not source_path.is_file():
            continue
        for audit in iter_jsonl(audit_path):
            if str(audit.get("reason")) != "final_observation_error:step_limit_exceeded":
                continue
            candidates += 1
            line_number = int(audit.get("line") or 0)
            row = None
            for index, candidate in enumerate(iter_jsonl(source_path), 1):
                if index == line_number:
                    row = candidate
                    break
            if row is None:
                rejected["source_line_missing"] += 1
                continue
            metadata = row.get("metadata") or {}
            source_index = int(metadata.get("source_index", -1))
            if source_index not in unrecovered:
                rejected["not_unrecovered"] += 1
                continue
            hard = by_index.get(source_index)
            if hard is None:
                rejected["hard_row_missing"] += 1
                continue
            messages = row.get("messages") or []
            prompt_messages = int(row.get("prompt_messages") or 0)
            errors = protocol_errors(messages, prompt_messages=prompt_messages)
            if errors:
                rejected["protocol:" + ",".join(errors)] += 1
                continue
            if has_meta_reference(messages, prompt_messages):
                rejected["meta_reference"] += 1
                continue
            grade_report = regrade(messages, hard, db_root=db_root)
            if grade_report.get("ex") is not True:
                rejected["offline_ex_failed"] += 1
                continue

            record = dict(row)
            new_metadata = dict(metadata)
            new_metadata.update(
                {
                    "recovery_tier": "B_budget",
                    "terminal_observation_kind": "step_limit_exceeded",
                    "budget_policy": "round2_accepts_budget_terminal",
                    "recovery_offline_ex": True,
                    "verification_mode": "offline_ex",
                    "ex_reason": grade_report.get("reason"),
                }
            )
            record["metadata"] = new_metadata
            record["data_variant"] = "hard_privileged_budget"
            previous = best.get(source_index)
            if previous is None or int(metadata.get("tool_calls") or 0) < int(
                (previous.get("metadata") or {}).get("tool_calls") or 0
            ):
                best[source_index] = record

    out = out_dir / "recovered_B_budget.jsonl"
    write_jsonl(out, best.values())
    return {
        "tier": "B_budget",
        "candidates": candidates,
        "recovered": len(best),
        "rejected": dict(rejected),
        "output": str(out),
    }


def _scan_attempts(
    *,
    raw_path: Path,
    round_name: str,
    unrecovered: set[int],
    by_index: Mapping[int, Mapping[str, Any]],
    best: Dict[int, Dict[str, Any]],
    rejected: Counter,
    db_root: Path,
) -> int:
    scanned = 0
    for attempt in iter_jsonl(raw_path):
        source_index = int(attempt.get("source_index", -1))
        if source_index not in unrecovered:
            continue
        issues = set((attempt.get("acceptance") or {}).get("issues") or [])
        if not issues or not issues <= ATTEMPT_ALLOWED_CODES:
            continue
        if (attempt.get("grade") or {}).get("ex") is not True:
            continue
        episode = attempt.get("episode") or {}
        transcript = episode.get("transcript") or []
        hard = by_index.get(source_index)
        if hard is None or not transcript:
            continue
        prompt_meta = attempt.get("prompt_meta") or {}
        prefix, reason = rebuild_deployment_prefix(hard, prompt_meta)
        if prefix is None:
            rejected["rebuild:" + str(reason)] += 1
            continue
        messages = append_transcript(prefix, transcript)
        if messages is None:
            rejected["bad_transcript_role"] += 1
            continue
        prompt_messages = len(prefix)
        scanned += 1

        tier = "C_pointer"
        if needs_pointer_repair(issues):
            repaired, ops, repair_reason = repair_verified_call(
                messages, prompt_messages=prompt_messages
            )
            if repaired is None:
                rejected[f"unrepairable:{repair_reason}"] += 1
                continue
            messages = repaired
        else:
            ops = []
            if executed_final_call(messages, prompt_messages=prompt_messages) is None:
                rejected["no_successful_exact_final_execution"] += 1
                continue
        if SINGLE_CALL_CODE in issues:
            tier = "C_single_call"

        remaining = protocol_errors(messages, prompt_messages=prompt_messages)
        if remaining:
            rejected["post_repair:" + ",".join(remaining)] += 1
            continue
        if has_meta_reference(messages, prompt_messages):
            rejected["meta_reference"] += 1
            continue
        grade_report = regrade(messages, hard, db_root=db_root)
        if grade_report.get("ex") is not True:
            rejected["offline_ex_failed"] += 1
            continue

        record = base_record(
            hard=hard,
            messages=messages,
            prompt_messages=prompt_messages,
            prompt_meta=prompt_meta,
            model=str(attempt.get("model") or ""),
            decoding=str(attempt.get("decoding") or ""),
            attempt=int(attempt.get("attempt") or 0),
            source_index=source_index,
            tier=tier,
        )
        record["metadata"].update(
            {
                "recovery_round": round_name,
                "original_issue_codes": sorted(issues),
                "repair_ops": ops,
                "tool_calls": (attempt.get("acceptance") or {}).get("tool_calls"),
                "assistant_turns": (attempt.get("acceptance") or {}).get("assistant_turns"),
                "ex_reason": grade_report.get("reason"),
                "verification_mode": "repaired" if ops and tier == "C_pointer" else "strict",
            }
        )
        previous = best.get(source_index)
        if previous is None or _rank(record) < _rank(previous):
            best[source_index] = record
    return scanned


def needs_pointer_repair(issues: Iterable[str]) -> bool:
    return bool(set(issues) & POINTER_CODES)


def _rank(row: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    metadata = row.get("metadata") or {}
    return (
        0 if metadata.get("recovery_tier") == "C_single_call" else 1,
        int(metadata.get("tool_calls") or 0),
        int(metadata.get("attempt") or 0),
        int(metadata.get("source_index") or 0),
    )


def recover_attempts(*, out_dir: Path, db_root: Path) -> Dict[str, Any]:
    """Tiers C_pointer and C_single_call, scanned from raw attempts."""

    unrecovered = unrecovered_indexes()
    by_index = load_hard_pool()
    best: Dict[int, Dict[str, Any]] = {}
    rejected: Counter = Counter()
    scanned = 0
    for branch in BRANCHES:
        raw = PRIV / branch / "raw_attempts.jsonl"
        if raw.is_file():
            scanned += _scan_attempts(
                raw_path=raw,
                round_name="round1",
                unrecovered=unrecovered,
                by_index=by_index,
                best=best,
                rejected=rejected,
                db_root=db_root,
            )
    for branch in BRANCHES:
        raw = ROUND2 / branch / "raw_attempts.jsonl"
        if raw.is_file():
            scanned += _scan_attempts(
                raw_path=raw,
                round_name="round2",
                unrecovered=unrecovered,
                by_index=by_index,
                best=best,
                rejected=rejected,
                db_root=db_root,
            )

    pointer = {k: v for k, v in best.items() if (v.get("metadata") or {}).get("recovery_tier") == "C_pointer"}
    single = {k: v for k, v in best.items() if (v.get("metadata") or {}).get("recovery_tier") == "C_single_call"}
    out_pointer = out_dir / "recovered_C_pointer.jsonl"
    out_single = out_dir / "recovered_C_single_call.jsonl"
    write_jsonl(out_pointer, pointer.values())
    write_jsonl(out_single, single.values())
    return {
        "tier": "C_pointer+C_single_call",
        "attempts_scanned": scanned,
        "recovered_C_pointer": len(pointer),
        "recovered_C_single_call": len(single),
        "rejected": dict(rejected),
        "outputs": [str(out_pointer), str(out_single)],
    }


def _unseen_gold_string_literals(
    messages: Sequence[Mapping[str, Any]],
    *,
    prompt_messages: int,
    gold_sql: str,
) -> List[str]:
    """Gold string literals the reasoning uses before they were visible.

    Mirrors ``audit_merged_sft``: the visible context is the question, the
    evidence, the M-Schema prompt, and every prior observation. A literal that
    appears in an assistant scratch pad before appearing in any of those came
    from the privileged reference, not from evidence.
    """

    from audit_merged_sft import sql_literals, text_has_literal, unwrap_observation  # noqa: E402

    strings, _ = sql_literals(gold_sql)
    if not strings:
        return []

    visible = "\n".join(
        [str(messages[prompt_messages - 1].get("content") or "") if prompt_messages else ""]
    )
    hits: List[str] = []
    for index in range(prompt_messages, len(messages)):
        message = messages[index]
        content = str(message.get("content") or "")
        if message.get("role") == "user":
            visible += "\n" + unwrap_observation(content)
            continue
        if message.get("role") != "assistant":
            continue
        check = content
        for literal in strings:
            core = str(literal).strip("%_")
            if len(core) < 2:
                continue
            if text_has_literal(check, str(literal)) and not text_has_literal(visible, str(literal)):
                if str(literal) not in hits:
                    hits.append(str(literal))
        visible += "\n" + content
    return hits


def recover_literal(*, out_dir: Path, db_root: Path) -> Dict[str, Any]:
    """Tier D: traces removed by the temporal gold-literal filter.

    The round-2 manifest records which samples were removed but not the literal
    detail, so the visibility check is recomputed here. Only samples whose
    unseen literals are all date/number scaffolding are re-admitted; a flagged
    answer entity stays out.
    """

    manifest_path = BACKUP / "literal_filter_manifest.json"
    backup_path = BACKUP / "sft_privileged_round2_merged.jsonl"
    if not manifest_path.is_file() or not backup_path.is_file():
        return {"tier": "D_literal_review", "recovered": 0, "skipped": "backup_missing"}

    unrecovered = unrecovered_indexes()
    by_index = load_hard_pool()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = set(manifest.get("target_sample_ids") or [])
    by_sample: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(backup_path):
        sample = str((row.get("metadata") or {}).get("sample_id") or "")
        if sample in targets:
            by_sample[sample] = row

    recovered: List[Dict[str, Any]] = []
    rejected: Counter = Counter()
    for row in by_sample.values():
        metadata = row.get("metadata") or {}
        source_index = int(metadata.get("source_index", -1))
        if source_index not in unrecovered:
            rejected["not_unrecovered"] += 1
            continue
        hard = by_index.get(source_index)
        if hard is None:
            rejected["hard_row_missing"] += 1
            continue
        messages = row.get("messages") or []
        prompt_messages = int(row.get("prompt_messages") or 0)
        hits = _unseen_gold_string_literals(
            messages,
            prompt_messages=prompt_messages,
            gold_sql=str(hard.get("SQL") or ""),
        )
        if not hits:
            rejected["no_unseen_literal_on_reaudit"] += 1
            continue
        non_benign = [literal for literal in hits if not BENIGN_LITERAL.match(literal.strip())]
        if non_benign:
            rejected["answer_entity_literal"] += 1
            continue
        errors = protocol_errors(messages, prompt_messages=prompt_messages)
        if errors:
            rejected["protocol:" + ",".join(errors)] += 1
            continue
        if has_meta_reference(messages, prompt_messages):
            rejected["meta_reference"] += 1
            continue
        grade_report = regrade(messages, hard, db_root=db_root)
        if grade_report.get("ex") is not True:
            rejected["offline_ex_failed"] += 1
            continue
        record = dict(row)
        new_metadata = dict(metadata)
        new_metadata.update(
            {
                "recovery_tier": "D_literal_review",
                "literal_review": "benign_format_only",
                "reviewed_literals": hits,
                "verification_mode": "strict",
                "ex_reason": grade_report.get("reason"),
            }
        )
        record["metadata"] = new_metadata
        recovered.append(record)

    out = out_dir / "recovered_D_literal.jsonl"
    write_jsonl(out, recovered)
    return {
        "tier": "D_literal_review",
        "targets": len(targets),
        "found_in_backup": len(by_sample),
        "recovered": len(recovered),
        "rejected": dict(rejected),
        "output": str(out),
    }


def clean_base(*, out: Path, summary: Path) -> Dict[str, Any]:
    """Validate base rows and annotate how each one ends.

    Two defects are dropped outright: a protocol error under the shared
    validator, and any mention of the privileged reference. Traces whose last
    call is a diagnostic rather than the final query are kept -- the shared
    protocol accepts them, the executed SQL was the final SQL, and EX is the
    authority -- but they are annotated so a strict-last-call ablation can drop
    them without re-parsing.
    """

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    reasons: Counter = Counter()
    last_call: Counter = Counter()
    for row in iter_jsonl(FINAL_2):
        messages = row.get("messages") or []
        prompt_messages = int(row.get("prompt_messages") or 0)
        errors = protocol_errors(messages, prompt_messages=prompt_messages)
        meta_leak = has_meta_reference(messages, prompt_messages)
        if errors or meta_leak:
            for error in errors:
                reasons[error] += 1
            if meta_leak:
                reasons["meta_reference"] += 1
            dropped.append(
                {
                    "id": row.get("id"),
                    "sample_id": (row.get("metadata") or {}).get("sample_id"),
                    "errors": errors or ["meta_reference"],
                }
            )
            continue
        annotated = dict(row)
        metadata = dict(row.get("metadata") or {})
        metadata["last_call_is_final"] = _last_call_is_final(messages, prompt_messages)
        metadata["verification_mode"] = "strict"
        annotated["metadata"] = metadata
        last_call[str(metadata["last_call_is_final"])] += 1
        kept.append(annotated)
    write_jsonl(out, kept)
    write_jsonl(summary.with_suffix(".dropped.jsonl"), dropped)
    payload = {
        "input": str(FINAL_2),
        "input_records": len(kept) + len(dropped),
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_reasons": dict(reasons),
        "last_call_is_final": dict(last_call),
        "output": str(out),
    }
    write_json(summary, payload)
    return payload


def _last_call_is_final(messages: Sequence[Mapping[str, Any]], prompt_messages: int) -> bool:
    from common.protocol import extract_final_sql, parse_tool_calls

    calls: List[Dict[str, Any]] = []
    for message in messages[prompt_messages:]:
        if message.get("role") == "assistant":
            calls.extend(parse_tool_calls(str(message.get("content") or "")))
    if not calls or calls[-1]["name"] != "sqlite_query":
        return False
    final_sql = extract_final_sql(messages, prompt_messages=prompt_messages)
    return B.normalize_sql(str(calls[-1]["arguments"].get("sql") or "")) == B.normalize_sql(
        str(final_sql or "")
    )


def merge_final(*, out_dir: Path, base: Path, out: Path, summary: Path) -> Dict[str, Any]:
    """Merge cleaned base and recovered tiers, one row per sample."""

    tiers: Dict[str, List[Dict[str, Any]]] = {"base_cleaned": list(iter_jsonl(base))}
    for name in (
        "recovered_B_budget.jsonl",
        "recovered_C_pointer.jsonl",
        "recovered_C_single_call.jsonl",
        "recovered_D_literal.jsonl",
    ):
        path = out_dir / name
        if path.is_file():
            tiers[name[: -len(".jsonl")]] = list(iter_jsonl(path))

    order = [
        "base_cleaned",
        "recovered_B_budget",
        "recovered_C_pointer",
        "recovered_C_single_call",
        "recovered_D_literal",
    ]
    chosen: Dict[str, Dict[str, Any]] = {}
    collisions = Counter()
    for tier in order:
        for row in tiers.get(tier, []):
            sample = str((row.get("metadata") or {}).get("sample_id") or "")
            if not sample:
                collisions["missing_sample_id"] += 1
                continue
            if sample in chosen:
                collisions[f"duplicate:{tier}"] += 1
                continue
            chosen[sample] = row

    rows = list(chosen.values())
    write_jsonl(out, rows)
    tier_counts = Counter(
        str((row.get("metadata") or {}).get("recovery_tier") or "base_cleaned") for row in rows
    )
    payload = {
        "policy": "one trajectory per sample; cleaned base first, then recovery tiers",
        "base": str(base),
        "base_records": len(tiers["base_cleaned"]),
        "inputs": {tier: len(rows_) for tier, rows_ in tiers.items()},
        "collisions": dict(collisions),
        "output_records": len(rows),
        "tier_counts": dict(tier_counts),
        "sample_ids_unique": len({str((row.get('metadata') or {}).get('sample_id')) for row in rows}),
        "output": str(out),
    }
    write_json(summary, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-root", type=Path, default=B.DEFAULT_DB_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("recover", help="run all recovery tiers")
    p.add_argument("--out-dir", type=Path, default=PRIV / "recovered")
    p = sub.add_parser("clean-base", help="drop base rows that fail the strict protocol")
    p.add_argument("--out", type=Path, default=PRIV / "final/sft_hard_base.jsonl")
    p.add_argument("--summary", type=Path, default=PRIV / "final/sft_hard_base.summary.json")
    p = sub.add_parser("merge", help="merge cleaned base and recovered tiers")
    p.add_argument("--out-dir", type=Path, default=PRIV / "recovered")
    p.add_argument("--base", type=Path, default=PRIV / "final/sft_hard_base.jsonl")
    p.add_argument("--out", type=Path, default=PRIV / "final/sft_hard_final.jsonl")
    p.add_argument("--summary", type=Path, default=PRIV / "final/sft_hard_final.summary.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "recover":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        reports = [
            recover_budget(out_dir=args.out_dir, db_root=args.db_root),
            recover_attempts(out_dir=args.out_dir, db_root=args.db_root),
            recover_literal(out_dir=args.out_dir, db_root=args.db_root),
        ]
        write_json(args.out_dir / "recovery.summary.json", {"tiers": reports})
        print(json.dumps({"tiers": reports}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "clean-base":
        print(json.dumps(clean_base(out=args.out, summary=args.summary), ensure_ascii=False, indent=2))
        return 0
    if args.command == "merge":
        print(
            json.dumps(
                merge_final(out_dir=args.out_dir, base=args.base, out=args.out, summary=args.summary),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
