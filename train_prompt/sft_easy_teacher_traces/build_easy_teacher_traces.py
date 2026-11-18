#!/usr/bin/env python3
"""Non-teacher-pool adapter for the frozen first-round trace builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

UPSTREAM = Path(
    "/data/keenSQL/train_prompt/sft_privileged_traces_kimi_ds/build_sft_traces.py"
)

spec = importlib.util.spec_from_file_location("_easy_teacher_upstream", UPSTREAM)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {UPSTREAM}")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

_original_make_sft_record = adapter.core.make_sft_record
_original_sample_id = adapter.core.sample_id


def stable_sample_id(source_index: int, record: dict[str, Any]) -> str:
    """Keep resume IDs stable after completed-only chunk compaction."""

    resume_index = record.get("resume_source_index")
    if resume_index is not None:
        source_index = int(resume_index)
    return _original_sample_id(source_index, record)


def make_easy_sft_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
    row = _original_make_sft_record(*args, **kwargs)
    record = kwargs.get("record")
    if not isinstance(record, dict):
        record = {}
    row["data_variant"] = "easy_teacher"
    row["source"] = record.get("source")
    row["difficulty"] = record.get("difficulty")
    row["sft_split"] = record.get("sft_split")
    row["current_easy_id"] = record.get("current_easy_id")
    metadata = row.setdefault("metadata", {})
    metadata["input_variant"] = record.get("input_variant") or "easy"
    metadata["original_decoding"] = record.get("original_decoding") or "pass16_temp1"
    metadata["original_teacher_model"] = (
        record.get("original_teacher_model") or "Qwen3.5-9B"
    )
    metadata["sft_split"] = record.get("sft_split")
    metadata["current_easy_id"] = record.get("current_easy_id")
    metadata["source_easy_id"] = record.get("source_easy_id")
    return row


adapter.core.make_sft_record = make_easy_sft_record
adapter.core.sample_id = stable_sample_id


if __name__ == "__main__":
    raise SystemExit(adapter.core.main())
