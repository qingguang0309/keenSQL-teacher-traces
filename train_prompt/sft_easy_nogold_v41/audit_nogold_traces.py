#!/usr/bin/env python3
"""Run the frozen first-round strict auditor for no-gold easy traces.

Identical to ../sft_easy_teacher_traces/audit_easy_teacher_traces.py except for
the tool-call range: the generator accepts a single call for no-gold traces
(build_nogold_traces.MIN_TOOL_CALLS), so the auditor's hard-coded 2-6 range is
widened to 1-6. A single-call row keeps a warning so it stays filterable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

UPSTREAM = Path(
    "/data/keenSQL/train_prompt/sft_privileged_traces/audit_merged_sft.py"
)
RANGE_ISSUE = "tool_call_count_outside_2_to_6"
SINGLE_CALL_WARNING = "single_tool_call_allowed_nogold"

spec = importlib.util.spec_from_file_location("_nogold_auditor", UPSTREAM)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {UPSTREAM}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

_original_audit_row = module.audit_row


def audit_row(index: int, row: dict[str, Any], db_root: Path) -> tuple[int, dict[str, Any]]:
    index, result = _original_audit_row(index, row, db_root)
    if result["evidence"].get("tool_calls") == 1 and RANGE_ISSUE in result["issues"]:
        result["issues"] = [issue for issue in result["issues"] if issue != RANGE_ISSUE]
        result["warnings"] = result["warnings"] + [SINGLE_CALL_WARNING]
        result["hard_pass"] = not result["issues"]
    return index, result


module.audit_row = audit_row


if __name__ == "__main__":
    raise SystemExit(module.main())
