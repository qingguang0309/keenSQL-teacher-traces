#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/data/keenSQL/.venv/bin/python}"
ROOT="/data/keenSQL/train_prompt/sft_privileged_traces"
SCRIPT="$ROOT/build_sft_traces.py"
INPUT="${SFT_HARD_INPUT:-/data/keenSQL/dataset/sft_data/data/all_dataset_qwen35_9b_pass16_full_pcache_bt16384_wrong.json}"
OUTPUT_ROOT="${SFT_TRACE_OUTPUT_ROOT:-/data/keenSQL/dataset/sft_data/privileged_traces}"
LOG_ROOT="/data/keenSQL/dataset/sft_data/logs"
SCHEMA_CACHE="/data/keenSQL/dataset/m_schema/cache"

usage() {
  echo "usage: $0 {qwen35-greedy|qwen35-pass8|qwen36plus-greedy|qwen36plus-pass8|status|merge|merge-qwen35|merge-qwen36plus|compare}"
}

launch() {
  local mode="$1"
  local model_url model decoding workers api_key
  case "$mode" in
    qwen35-greedy)
      model="Qwen3.5-9B"; model_url="http://127.0.0.1:8000/v1"
      decoding="greedy"; workers="${SFT_LOCAL_WORKERS:-32}"; api_key="EMPTY"
      ;;
    qwen35-pass8)
      model="Qwen3.5-9B"; model_url="http://127.0.0.1:8000/v1"
      decoding="pass8"; workers="${SFT_LOCAL_WORKERS:-32}"; api_key="EMPTY"
      ;;
    qwen36plus-greedy)
      model="qwen3.6-plus"; model_url="${QWEN_BASE_URL:?export QWEN_BASE_URL first}"
      decoding="greedy"; workers="${SFT_API_WORKERS:-8}"; api_key="${QWEN_API_KEY:?export QWEN_API_KEY first}"
      ;;
    qwen36plus-pass8)
      model="qwen3.6-plus"; model_url="${QWEN_BASE_URL:?export QWEN_BASE_URL first}"
      decoding="pass8"; workers="${SFT_API_WORKERS:-8}"; api_key="${QWEN_API_KEY:?export QWEN_API_KEY first}"
      ;;
    *) usage; exit 2 ;;
  esac

  mkdir -p "$OUTPUT_ROOT/$mode" "$LOG_ROOT"
  local log="$LOG_ROOT/sft_privileged_${mode}.log"
  local pid_file="$LOG_ROOT/sft_privileged_${mode}.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$mode is already running with PID $(cat "$pid_file")"
    exit 1
  fi

  QWEN_API_KEY="$api_key" nohup "$PY" "$SCRIPT" generate \
    --input "$INPUT" \
    --schema-cache "$SCHEMA_CACHE" \
    --output-dir "$OUTPUT_ROOT/$mode" \
    --model-url "$model_url" \
    --model "$model" \
    --decoding "$decoding" \
    --workers "$workers" \
    --max-tokens 8192 \
    --timeout 600 \
    --max-turns 7 \
    --max-prompt-tokens 16000 \
    --seed 0 \
    >>"$log" 2>&1 &
  echo $! >"$pid_file"
  echo "started $mode pid=$(cat "$pid_file") log=$log output=$OUTPUT_ROOT/$mode"
}

status() {
  local mode pid_file pid state
  for mode in qwen35-greedy qwen35-pass8 qwen36plus-greedy qwen36plus-pass8; do
    pid_file="$LOG_ROOT/sft_privileged_${mode}.pid"
    if [[ -s "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      state="stopped"
      kill -0 "$pid" 2>/dev/null && state="running"
      echo "$mode pid=$pid state=$state log=$LOG_ROOT/sft_privileged_${mode}.log"
    else
      echo "$mode not started"
    fi
  done
}

merge_group() {
  local group="$1"
  shift
  local inputs=()
  for mode in "$@"; do
    local path="$OUTPUT_ROOT/$mode/accepted_sft.jsonl"
    if [[ -f "$path" ]]; then
      inputs+=("$path")
    else
      echo "skip missing $path" >&2
    fi
  done
  if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "no accepted files for $group" >&2
    exit 1
  fi
  "$PY" "$SCRIPT" merge \
    --inputs "${inputs[@]}" \
    --out "$OUTPUT_ROOT/sft_privileged_${group}.jsonl" \
    --summary "$OUTPUT_ROOT/sft_privileged_${group}.summary.json"
}

merge_all() {
  local inputs=()
  for mode in qwen35-greedy qwen35-pass8 qwen36plus-greedy qwen36plus-pass8; do
    local path="$OUTPUT_ROOT/$mode/accepted_sft.jsonl"
    [[ -f "$path" ]] && inputs+=("$path")
  done
  if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "no accepted files under $OUTPUT_ROOT" >&2
    exit 1
  fi
  "$PY" "$SCRIPT" merge \
    --inputs "${inputs[@]}" \
    --out "$OUTPUT_ROOT/sft_privileged_merged.jsonl" \
    --summary "$OUTPUT_ROOT/sft_privileged_merged.summary.json"
}

compare_groups() {
  "$PY" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for mode in (
    "qwen35-greedy",
    "qwen35-pass8",
    "qwen36plus-greedy",
    "qwen36plus-pass8",
):
    summary_path = root / mode / "summary.json"
    if not summary_path.is_file():
        continue
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "mode": mode,
            "attempts": payload.get("attempts"),
            "accepted": payload.get("accepted"),
            "acceptance_rate": payload.get("acceptance_rate"),
            "samples_with_accepted_trace": payload.get("samples_with_accepted_trace"),
            "issues": payload.get("issues"),
            "warnings": payload.get("warnings"),
        }
    )
for group in ("qwen35", "qwen36plus", "merged"):
    summary_path = root / f"sft_privileged_{group}.summary.json"
    if not summary_path.is_file():
        continue
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "mode": f"merged:{group}",
            "records": payload.get("merged_records"),
            "duplicates_by_input": payload.get("duplicates_by_input"),
        }
    )
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
}

case "${1:-}" in
  qwen35-greedy|qwen35-pass8|qwen36plus-greedy|qwen36plus-pass8) launch "$1" ;;
  status) status ;;
  merge) merge_all ;;
  merge-qwen35) shift; merge_group qwen35 qwen35-greedy qwen35-pass8 ;;
  merge-qwen36plus) shift; merge_group qwen36plus qwen36plus-greedy qwen36plus-pass8 ;;
  compare) compare_groups ;;
  *) usage; exit 2 ;;
esac
