#!/usr/bin/env python3
"""Run the frozen first-round strict auditor for regenerated easy traces."""

from __future__ import annotations

import importlib.util
from pathlib import Path

UPSTREAM = Path(
    "/data/keenSQL/train_prompt/sft_privileged_traces/audit_merged_sft.py"
)

spec = importlib.util.spec_from_file_location("_easy_teacher_auditor", UPSTREAM)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {UPSTREAM}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    raise SystemExit(module.main())
