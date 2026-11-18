#!/usr/bin/env bash
# End-to-end check of trace_pipeline_v1 on a few questions (needs OPENROUTER_API_KEY in env).
# usage: examples_smoke_e2e.sh ROOT IDS_FILE [TEACHER]
set -euo pipefail
ROOT=$1; IDS=$2; TEACHER=${3:-openai/gpt-5.6-sol}
PY=/data/keenSQL/.venv/bin/python
PL="$PY /data/keenSQL/train_prompt/trace_pipeline_v1/pipeline.py"
PARENT=/data/keenSQL/dataset/m_schema/sft_data/easy_nogold_v41_pilot500_gpt56sol/inputs/easy_current.json
$PL prepare --root "$ROOT" --parent-input "$PARENT" --ids-file "$IDS"
$PL gate --root "$ROOT" --teacher "$TEACHER" --budget 5
$PL smoke --root "$ROOT" --teacher "$TEACHER" --n 3
$PL generate --root "$ROOT" --teacher "$TEACHER" --workers 4 --guard-interval 30 --wait
$PL merge --root "$ROOT" --teacher "$TEACHER"
$PL audit --root "$ROOT" --teacher "$TEACHER" --run-name smoke8 --out "$ROOT/audit"
$PL release --audit "$ROOT/audit" --gen-root "$ROOT" --out "$ROOT/release"
echo E2E_DONE
