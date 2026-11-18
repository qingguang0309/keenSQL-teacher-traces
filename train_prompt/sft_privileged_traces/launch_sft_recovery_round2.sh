#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/data/keenSQL/.venv/bin/python}"
SCRIPT_ROOT="/data/keenSQL/train_prompt/sft_privileged_traces"
BUILD="$SCRIPT_ROOT/build_sft_recovery_round2.py"
PREPARE="$SCRIPT_ROOT/prepare_sft_recovery_round2.py"
COMBINE="$SCRIPT_ROOT/combine_sft_recovery_round2.py"
FINAL_SUCCESS_FILTER="$SCRIPT_ROOT/filter_successful_final_execution.py"
TEACHER_PROMPT="${SFT_RECOVERY_TEACHER_PROMPT:-/data/keenSQL/train_prompt/retoolsql_prompt/v4_recovery_round2/sft_teacher_prompt.txt}"
HARD_INPUT="${SFT_HARD_INPUT:-/data/keenSQL/dataset/sft_data/data/all_dataset_qwen35_9b_pass16_full_pcache_bt16384_wrong.json}"
BASE_MERGED="${SFT_BASE_MERGED:-/data/keenSQL/dataset/sft_data/privileged_traces/sft_privileged_merged.jsonl}"
OUTPUT_ROOT="${SFT_RECOVERY_OUTPUT_ROOT:-/data/keenSQL/dataset/sft_data/privileged_traces/round2_recovery}"
INPUT="$OUTPUT_ROOT/input/unmerged_hard_samples.json"
INPUT_MANIFEST="$OUTPUT_ROOT/input/manifest.json"
LOG_ROOT="$OUTPUT_ROOT/logs"
MERGE_ROOT="$OUTPUT_ROOT/merged"
SCHEMA_CACHE="/data/keenSQL/dataset/m_schema/cache"

MODES=(qwen35-greedy qwen35-pass8 qwen36plus-greedy qwen36plus-pass8)

usage() {
  echo "usage: $0 {prepare|qwen35-greedy|qwen35-pass8|qwen36plus-greedy|qwen36plus-pass8|greedy-pair|pass8-pair|status|merge-round2|merge-final|merge-all|compare}"
}

prepare_input() {
  mkdir -p "$OUTPUT_ROOT/input" "$LOG_ROOT" "$MERGE_ROOT"
  "$PY" "$PREPARE" \
    --hard-input "$HARD_INPUT" \
    --base-merged "$BASE_MERGED" \
    --out "$INPUT" \
    --manifest "$INPUT_MANIFEST"
}

require_input() {
  if [[ ! -s "$INPUT" || ! -s "$INPUT_MANIFEST" ]]; then
    echo "missing recovery input; run: $0 prepare" >&2
    return 1
  fi
}

require_qwen_api() {
  : "${QWEN_BASE_URL:?export QWEN_BASE_URL first}"
  : "${QWEN_API_KEY:?export QWEN_API_KEY first}"
}

launch() {
  local mode="$1"
  local model_url model decoding workers api_key
  require_input
  case "$mode" in
    qwen35-greedy)
      model="Qwen3.5-9B"; model_url="${SFT_LOCAL_MODEL_URL:-http://127.0.0.1:8000/v1}"
      decoding="greedy"; workers="${SFT_RECOVERY_LOCAL_WORKERS:-8}"; api_key="EMPTY"
      ;;
    qwen35-pass8)
      model="Qwen3.5-9B"; model_url="${SFT_LOCAL_MODEL_URL:-http://127.0.0.1:8000/v1}"
      decoding="pass8"; workers="${SFT_RECOVERY_LOCAL_WORKERS:-8}"; api_key="EMPTY"
      ;;
    qwen36plus-greedy)
      require_qwen_api
      model="qwen3.6-plus"; model_url="$QWEN_BASE_URL"
      decoding="greedy"; workers="${SFT_RECOVERY_API_WORKERS:-4}"; api_key="$QWEN_API_KEY"
      ;;
    qwen36plus-pass8)
      require_qwen_api
      model="qwen3.6-plus"; model_url="$QWEN_BASE_URL"
      decoding="pass8"; workers="${SFT_RECOVERY_API_WORKERS:-4}"; api_key="$QWEN_API_KEY"
      ;;
    *) usage; return 2 ;;
  esac

  mkdir -p "$OUTPUT_ROOT/$mode" "$LOG_ROOT"
  local log="$LOG_ROOT/$mode.log"
  local pid_file="$LOG_ROOT/$mode.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$mode is already running with PID $(cat "$pid_file")"
    return 0
  fi

  echo "[$(date -Is)] starting $mode" >>"$log"
  QWEN_API_KEY="$api_key" nohup "$PY" "$BUILD" generate \
    --input "$INPUT" \
    --schema-cache "$SCHEMA_CACHE" \
    --teacher-prompt "$TEACHER_PROMPT" \
    --output-dir "$OUTPUT_ROOT/$mode" \
    --model-url "$model_url" \
    --model "$model" \
    --decoding "$decoding" \
    --workers "$workers" \
    --max-tokens 8192 \
    --timeout 600 \
    --max-turns 9 \
    --max-prompt-tokens 16000 \
    --seed 0 \
    >>"$log" 2>&1 &
  echo $! >"$pid_file"
  echo "started $mode pid=$(cat "$pid_file") log=$log output=$OUTPUT_ROOT/$mode"
}

status() {
  local mode pid_file pid state summary
  for mode in "${MODES[@]}"; do
    pid_file="$LOG_ROOT/$mode.pid"
    summary="$OUTPUT_ROOT/$mode/summary.json"
    if [[ -s "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      state="stopped"
      kill -0 "$pid" 2>/dev/null && state="running"
      echo "$mode pid=$pid state=$state log=$LOG_ROOT/$mode.log"
    else
      echo "$mode not started"
    fi
    if [[ -s "$summary" ]]; then
      "$PY" - "$summary" <<'PY'
import json
import sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"  attempts={p.get('attempts')} accepted={p.get('accepted')} samples={p.get('samples_with_accepted_trace')}")
PY
    fi
  done
}

merge_round2() {
  local inputs=()
  local mode path filtered audit filter_summary
  mkdir -p "$MERGE_ROOT"
  for mode in "${MODES[@]}"; do
    path="$OUTPUT_ROOT/$mode/accepted_sft.jsonl"
    if [[ ! -f "$path" ]]; then
      echo "cannot merge: missing $path" >&2
      return 1
    fi
    filtered="$OUTPUT_ROOT/$mode/accepted_sft.final_success.jsonl"
    audit="$OUTPUT_ROOT/$mode/accepted_sft.final_success.rejected.jsonl"
    filter_summary="$OUTPUT_ROOT/$mode/accepted_sft.final_success.summary.json"
    "$PY" "$FINAL_SUCCESS_FILTER" \
      --input "$path" \
      --output "$filtered" \
      --audit "$audit" \
      --summary "$filter_summary"
    inputs+=("$filtered")
  done
  "$PY" "$BUILD" merge \
    --inputs "${inputs[@]}" \
    --out "$MERGE_ROOT/sft_privileged_round2_merged.jsonl" \
    --summary "$MERGE_ROOT/sft_privileged_round2_merged.summary.json"
}

merge_final() {
  local recovery="$MERGE_ROOT/sft_privileged_round2_merged.jsonl"
  if [[ ! -s "$recovery" ]]; then
    echo "missing round-two merge; run: $0 merge-round2" >&2
    return 1
  fi
  "$PY" "$COMBINE" \
    --base "$BASE_MERGED" \
    --recovery "$recovery" \
    --manifest "$INPUT_MANIFEST" \
    --out "$MERGE_ROOT/sft_privileged_combined_round2.jsonl" \
    --summary "$MERGE_ROOT/sft_privileged_combined_round2.summary.json"
}

compare_runs() {
  "$PY" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for mode in ("qwen35-greedy", "qwen35-pass8", "qwen36plus-greedy", "qwen36plus-pass8"):
    path = root / mode / "summary.json"
    if path.is_file():
        p = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "mode": mode,
            "attempts": p.get("attempts"),
            "accepted": p.get("accepted"),
            "samples_with_accepted_trace": p.get("samples_with_accepted_trace"),
            "issue_counts": p.get("issue_counts"),
        })
for name in ("sft_privileged_round2_merged", "sft_privileged_combined_round2"):
    path = root / "merged" / f"{name}.summary.json"
    if path.is_file():
        rows.append({"mode": name, **json.loads(path.read_text(encoding="utf-8"))})
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
}

case "${1:-}" in
  prepare) prepare_input ;;
  qwen35-greedy|qwen35-pass8|qwen36plus-greedy|qwen36plus-pass8) launch "$1" ;;
  greedy-pair) require_qwen_api; launch qwen35-greedy; launch qwen36plus-greedy ;;
  pass8-pair) require_qwen_api; launch qwen35-pass8; launch qwen36plus-pass8 ;;
  status) status ;;
  merge-round2) merge_round2 ;;
  merge-final) merge_final ;;
  merge-all) merge_round2; merge_final ;;
  compare) compare_runs ;;
  *) usage; exit 2 ;;
esac
