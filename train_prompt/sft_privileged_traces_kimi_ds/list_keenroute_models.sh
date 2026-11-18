#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/data/keenSQL/.venv/bin/python}"
BASE_URL="${KEENROUTE_BASE_URL:?set KEENROUTE_BASE_URL to the KeenRoute OpenAI-compatible base URL}"

if [[ -z "${KEENROUTE_API_KEY:-}" ]]; then
  echo "missing KEENROUTE_API_KEY; run: export KEENROUTE_API_KEY='your-key'" >&2
  exit 2
fi

MODEL_LIST_API_KEY="$KEENROUTE_API_KEY" "$PY" - "$BASE_URL" <<'PY'
import os
import sys

from openai import OpenAI

base_url = sys.argv[1]
try:
    client = OpenAI(
        api_key=os.environ["MODEL_LIST_API_KEY"],
        base_url=base_url,
        timeout=60,
        max_retries=2,
    )
    models = sorted({item.id for item in client.models.list().data if item.id})
except Exception as exc:
    print(f"KeenRoute model-list request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"provider=keenroute base_url={base_url} available_models={len(models)}")
for model in models:
    print(model)
PY
