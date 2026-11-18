#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/data/keenSQL/.venv/bin/python}"
SCRIPT_ROOT="/data/keenSQL/train_prompt/sft_privileged_traces"
OUTPUT_ROOT="${SFT_RECOVERY_OUTPUT_ROOT:-/data/keenSQL/dataset/sft_data/privileged_traces/round2_recovery}"
MERGE_ROOT="$OUTPUT_ROOT/merged"
BASE_MERGED="${SFT_BASE_MERGED:-/data/keenSQL/dataset/sft_data/privileged_traces/sft_privileged_merged.jsonl}"
INPUT_MANIFEST="$OUTPUT_ROOT/input/manifest.json"

LAUNCHER="$SCRIPT_ROOT/launch_sft_recovery_round2.sh"
FINAL_SUCCESS_FILTER="$SCRIPT_ROOT/filter_successful_final_execution.py"
TOOL_COUNT_FILTER="$SCRIPT_ROOT/filter_sft_by_tool_count.py"
BUILD="$SCRIPT_ROOT/build_sft_recovery_round2.py"
COMBINE="$SCRIPT_ROOT/combine_sft_recovery_round2.py"
MODES=(qwen35-greedy qwen35-pass8 qwen36plus-greedy qwen36plus-pass8)

usage() {
  echo "usage: $0 {default|multi-tool-only|all|report}"
}

require_finished_runs() {
  local mode pid_file pid
  for mode in "${MODES[@]}"; do
    pid_file="$OUTPUT_ROOT/logs/$mode.pid"
    if [[ -s "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "cannot merge while $mode is running (PID $pid)" >&2
        return 1
      fi
    fi
    if [[ ! -s "$OUTPUT_ROOT/$mode/accepted_sft.jsonl" ]]; then
      echo "cannot merge: missing $OUTPUT_ROOT/$mode/accepted_sft.jsonl" >&2
      return 1
    fi
  done
}

merge_default() {
  require_finished_runs
  "$LAUNCHER" merge-all
}

merge_multi_tool_only() {
  require_finished_runs
  mkdir -p "$MERGE_ROOT"
  local mode accepted final_success filtered audit summary
  local inputs=()
  for mode in "${MODES[@]}"; do
    accepted="$OUTPUT_ROOT/$mode/accepted_sft.jsonl"
    final_success="$OUTPUT_ROOT/$mode/accepted_sft.final_success.jsonl"
    "$PY" "$FINAL_SUCCESS_FILTER" \
      --input "$accepted" \
      --output "$final_success" \
      --audit "$OUTPUT_ROOT/$mode/accepted_sft.final_success.rejected.jsonl" \
      --summary "$OUTPUT_ROOT/$mode/accepted_sft.final_success.summary.json"

    filtered="$MERGE_ROOT/$mode.final_success.multi_tool_only.jsonl"
    audit="$MERGE_ROOT/$mode.final_success.multi_tool_only.rejected.jsonl"
    summary="$MERGE_ROOT/$mode.final_success.multi_tool_only.summary.json"
    "$PY" "$TOOL_COUNT_FILTER" \
      --input "$final_success" \
      --output "$filtered" \
      --audit "$audit" \
      --summary "$summary" \
      --min-tool-calls 2
    inputs+=("$filtered")
  done

  "$PY" "$BUILD" merge \
    --inputs "${inputs[@]}" \
    --out "$MERGE_ROOT/sft_privileged_round2_merged_multi_tool_only.jsonl" \
    --summary "$MERGE_ROOT/sft_privileged_round2_merged_multi_tool_only.summary.json"

  "$PY" "$COMBINE" \
    --base "$BASE_MERGED" \
    --recovery "$MERGE_ROOT/sft_privileged_round2_merged_multi_tool_only.jsonl" \
    --manifest "$INPUT_MANIFEST" \
    --out "$MERGE_ROOT/sft_privileged_combined_round2_multi_tool_only.jsonl" \
    --summary "$MERGE_ROOT/sft_privileged_combined_round2_multi_tool_only.summary.json"
}

report() {
  "$PY" - "$MERGE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = (
    "sft_privileged_round2_merged.summary.json",
    "sft_privileged_combined_round2.summary.json",
    "sft_privileged_round2_merged_multi_tool_only.summary.json",
    "sft_privileged_combined_round2_multi_tool_only.summary.json",
)
for name in names:
    path = root / name
    print(f"\n=== {name} ===")
    if path.is_file():
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    else:
        print("not generated")
PY
}

case "${1:-}" in
  default) merge_default ;;
  multi-tool-only) merge_multi_tool_only ;;
  all) merge_default; merge_multi_tool_only ;;
  report) report ;;
  *) usage; exit 2 ;;
esac
