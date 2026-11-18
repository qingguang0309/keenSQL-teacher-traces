#!/usr/bin/env python3
"""Run the original strict trajectory auditor from the isolated Kimi/DS job."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

UPSTREAM = Path(
    "/data/keenSQL/train_prompt/sft_privileged_traces/audit_merged_sft.py"
)
EXPECTED_SHA256 = "8016835fc6fdc9f01bf06d89fb687958a853cc962f63602f13e157dca782521b"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not UPSTREAM.is_file():
    raise FileNotFoundError(f"missing first-round auditor: {UPSTREAM}")
actual_hash = file_sha256(UPSTREAM)
if actual_hash != EXPECTED_SHA256 and os.environ.get("ALLOW_UPSTREAM_DRIFT") != "1":
    raise RuntimeError(
        "first-round auditor changed; refusing silent behavior drift: "
        f"expected={EXPECTED_SHA256} actual={actual_hash}"
    )

spec = importlib.util.spec_from_file_location("_sft_first_round_audit", UPSTREAM)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {UPSTREAM}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    raise SystemExit(module.main())
