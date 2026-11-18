"""Shared helpers for trace_pipeline_v1 (no third-party imports except openai, lazily)."""

from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path("/data/keenSQL")
HERE = Path(__file__).resolve().parent
# SFT_DB_ROOT / SFT_SCHEMA_CACHE select another database set (e.g. BIRD dev, whose formula_1 differs
# from the training formula_1 of the same name). Defaults are the training set.
DB_ROOT = Path(os.environ.get("SFT_DB_ROOT") or ROOT / "dataset/all_dataset/databases")
SCHEMA_CACHE = Path(os.environ.get("SFT_SCHEMA_CACHE") or ROOT / "dataset/m_schema/cache")
DEV_JSON = ROOT / "dataset/bird-dev/dev.json"
LAUNCHER = HERE / "launch_jobs_v2.sh"
V41_AUDITOR = ROOT / "train_prompt/sft_easy_nogold_v41/audit_nogold_traces.py"
OPENROUTER = "https://openrouter.ai/api/v1"

# Model families: a trace is never judged by its own teacher's family (decision 1).
FAMILY_ALIASES = {"moonshotai": "kimi", "kimi": "kimi", "deepseek": "deepseek",
                  "openai": "openai", "qwen": "qwen", "anthropic": "anthropic",
                  "google": "google", "x-ai": "xai", "z-ai": "glm", "mistralai": "mistral"}


def family(model: str) -> str:
    head = model.split("/", 1)[0].lower() if "/" in model else model.split("-", 1)[0].lower()
    return FAMILY_ALIASES.get(head, head)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path | str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    tmp.replace(path)
    return count


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def model_slug(model: str) -> str:
    """Same rule as the launcher's model_slug()."""
    import re
    return re.sub(r"[^a-z0-9._-]+", "-", model.strip().lower()).strip("-.")


def raw_attempt_files(output_root: Path | str) -> list[Path]:
    return sorted(Path(output_root).glob("models/*/*/chunk_*/raw_attempts.jsonl"))


def prepare_inputs(records: list[dict[str, Any]], out_root: Path, *, selection: str,
                   parent_input: Path | None, chunk_size: int = 34) -> dict[str, Any]:
    """Write <out_root>/inputs/{easy_current.json, chunks/, manifest.json} in the layout
    the launcher and verify_setup.py expect."""

    ids = [str(r["question_id"]) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question_id in selection")
    inputs = out_root / "inputs"
    chunks_dir = inputs / "chunks"
    if inputs.exists() and any(inputs.iterdir()):
        raise FileExistsError(f"{inputs} already exists; refusing to overwrite")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_path = inputs / "easy_current.json"
    all_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    chunk_records = []
    for index in range(0, len(records), chunk_size):
        part = records[index: index + chunk_size]
        name = f"chunk_{index // chunk_size:03d}"
        path = chunks_dir / f"{name}.json"
        path.write_text(json.dumps(part, ensure_ascii=False, indent=2), encoding="utf-8")
        chunk_records.append({"chunk": name, "path": str(path), "records": len(part),
                              "sha256": sha256_file(path),
                              "first_question_id": str(part[0]["question_id"]),
                              "last_question_id": str(part[-1]["question_id"])})
    manifest = {
        "created_at_utc": utc_now(),
        "selection": selection,
        "parent_input": str(parent_input) if parent_input else None,
        "parent_input_sha256": sha256_file(parent_input) if parent_input else None,
        "selected_records": len(records),
        "unique_question_ids": len(set(ids)),
        "all_input": str(all_path),
        "all_input_sha256": sha256_file(all_path),
        "chunks": len(chunk_records),
        "chunk_records": chunk_records,
        "generator": "trace_pipeline_v1/build_traces_v2.py",
    }
    write_json(inputs / "manifest.json", manifest)
    return manifest


def stratified_sample(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic sample spread over (source, difficulty) groups."""
    if n >= len(records):
        return list(records)
    rng = random.Random(seed)
    groups: dict[tuple, list] = {}
    for r in records:
        groups.setdefault((r.get("source"), r.get("difficulty")), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    picked: list[dict[str, Any]] = []
    keys = sorted(groups, key=str)
    while len(picked) < n:
        for key in keys:
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())
    return picked


def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return key


def http_json(url: str, key: str | None = None, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def openrouter_balance(key: str) -> dict[str, Any]:
    """Spendable USD = min(key limit remaining, account credits remaining).

    Both bind: the key has its own hard limit (hit on 2026-11-16 at $150), and the account's
    credits are shared with every other key on it (on 2026-11-16 the key had $197.65 left while
    the account had $145.50). `limit_remaining` is the effective value callers compare to floors.
    """
    data = http_json(f"{OPENROUTER}/key", key).get("data", {})
    key_remaining = data.get("limit_remaining")
    account_remaining = None
    try:
        credits = http_json(f"{OPENROUTER}/credits", key).get("data", {})
        account_remaining = float(credits["total_credits"]) - float(credits["total_usage"])
    except Exception:  # noqa: BLE001 - endpoint unavailable for this key: fall back to the key limit
        pass
    values = [float(v) for v in (key_remaining, account_remaining) if v is not None]
    return {"usage": data.get("usage"), "limit": data.get("limit"),
            "key_remaining": key_remaining, "account_remaining": account_remaining,
            "limit_remaining": min(values) if values else None}


def openrouter_models() -> dict[str, dict[str, Any]]:
    return {m["id"]: m for m in http_json(f"{OPENROUTER}/models").get("data", [])}
