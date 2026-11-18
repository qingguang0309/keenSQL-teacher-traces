#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/data/keenSQL/.venv/bin/python}"
ROOT="/data/keenSQL/train_prompt/sft_privileged_traces_kimi_ds"
SCRIPT="$ROOT/build_sft_traces.py"
AUDITOR="$ROOT/audit_merged_sft.py"
VERIFY="$ROOT/verify_setup.py"
PROMPT="$ROOT/sft_teacher_prompt_v1.txt"
INPUT="/data/keenSQL/dataset/sft_data/privileged_traces/other_teacher_input/remaining_hard_1382.json"
OUTPUT_ROOT="${SFT_KIMI_DS_OUTPUT_ROOT:-/data/keenSQL/dataset/sft_data/privileged_traces_sft_privileged_traces_kimi_ds}"
MODEL_ROOT="$OUTPUT_ROOT/models"
MERGED_ROOT="$OUTPUT_ROOT/merged"
LOG_ROOT="$OUTPUT_ROOT/logs"
DB_ROOT="/data/keenSQL/dataset/all_dataset/databases"
SCHEMA_CACHE="/data/keenSQL/dataset/m_schema/cache"

usage() {
  cat <<'EOF'
usage:
  launch_sft_jobs.sh verify
  launch_sft_jobs.sh check-api MODEL
  launch_sft_jobs.sh preflight MODEL {greedy|pass8}
  launch_sft_jobs.sh dry-run MODEL {greedy|pass8}
  launch_sft_jobs.sh start MODEL {greedy|pass8}
  launch_sft_jobs.sh status
  launch_sft_jobs.sh tail MODEL {greedy|pass8}
  launch_sft_jobs.sh merge-model MODEL
  launch_sft_jobs.sh merge-all
  launch_sft_jobs.sh compare

Required key for start/check-api/preflight:
  KIMI_API_KEY, DEEPSEEK_API_KEY, QWEN_API_KEY, or KEENROUTE_API_KEY
  (TEACHER_API_KEY is a generic fallback)
EOF
}

model_slug() {
  "$PY" - "$1" <<'PY'
import re, sys
slug = re.sub(r"[^a-z0-9._-]+", "-", sys.argv[1].strip().lower()).strip("-.")
if not slug:
    raise SystemExit("model name does not produce a safe directory name")
print(slug)
PY
}

provider_for_model() {
  local lower override
  override="${SFT_PROVIDER_OVERRIDE:-}"
  if [[ -n "$override" ]]; then
    case "$override" in
      deepseek|kimi|qwen|keenroute) printf '%s\n' "$override"; return 0 ;;
      *) echo "unsupported SFT_PROVIDER_OVERRIDE: $override" >&2; return 2 ;;
    esac
  fi
  lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$lower" in
    *deepseek*) echo deepseek ;;
    *kimi*|*moonshot*) echo kimi ;;
    *qwen*) echo qwen ;;
    gpt-*|*astra*|*claude*|*gemini*) echo keenroute ;;
    *)
      echo "cannot infer provider from model '$1'; set SFT_PROVIDER_OVERRIDE=keenroute for another relay model" >&2
      return 2
      ;;
  esac
}

base_url_for_provider() {
  case "$1" in
    deepseek) printf '%s\n' "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}" ;;
    kimi)
      if [[ -n "${KIMI_BASE_URL:-}" ]]; then
        printf '%s\n' "$KIMI_BASE_URL"
      elif [[ "${KIMI_REGION:-cn}" == "global" ]]; then
        printf '%s\n' "https://api.moonshot.ai/v1"
      else
        printf '%s\n' "https://api.moonshot.cn/v1"
      fi
      ;;
    qwen)
      printf '%s\n' "${QWEN_BASE_URL:-https://maas.qianwenaiapi.com/compatible-mode/v1}"
      ;;
    keenroute)
      printf '%s\n' "${KEENROUTE_BASE_URL:?set KEENROUTE_BASE_URL to the KeenRoute OpenAI-compatible base URL}"
      ;;
    *) return 2 ;;
  esac
}

key_for_provider() {
  local key=""
  case "$1" in
    deepseek) key="${DEEPSEEK_API_KEY:-${TEACHER_API_KEY:-}}" ;;
    kimi) key="${KIMI_API_KEY:-${MOONSHOT_API_KEY:-${TEACHER_API_KEY:-}}}" ;;
    qwen) key="${QWEN_API_KEY:-${DASHSCOPE_API_KEY:-${TEACHER_API_KEY:-}}}" ;;
    keenroute) key="${KEENROUTE_API_KEY:-${TEACHER_API_KEY:-}}" ;;
    *) return 2 ;;
  esac
  if [[ -z "$key" ]]; then
    echo "missing API key for $1" >&2
    return 2
  fi
  printf '%s\n' "$key"
}

thinking_mode_for_provider() {
  local provider="$1" model="${2:-}"
  case "$provider" in
    # DeepSeek thinking mode ignores temperature. Disable it so greedy=0 and
    # pass@8=1 preserve the first-round decoding definitions.
    deepseek) printf '%s\n' "${DEEPSEEK_THINKING_MODE:-disabled}" ;;
    # K2.6 defaults to thinking enabled at the provider. Use non-thinking mode
    # for this sampling job unless the caller explicitly overrides it.
    kimi)
      if [[ -n "${KIMI_THINKING_MODE+x}" ]]; then
        printf '%s\n' "$KIMI_THINKING_MODE"
      elif [[ "$model" == "kimi-k2.6" ]]; then
        printf '%s\n' "disabled"
      else
        printf '%s\n' "omit"
      fi
      ;;
    # Match the first-round Qwen audit path: do not inject provider-specific
    # thinking switches unless the caller explicitly asks for one.
    qwen) printf '%s\n' "${QWEN_THINKING_MODE:-omit}" ;;
    # The relay may route several API families. Keep the OpenAI-compatible
    # request minimal unless an explicit switch is known to be supported.
    keenroute) printf '%s\n' "${KEENROUTE_THINKING_MODE:-omit}" ;;
    *) return 2 ;;
  esac
}

reasoning_effort_for_provider() {
  case "$1" in
    deepseek) printf '%s\n' "${DEEPSEEK_REASONING_EFFORT:-}" ;;
    kimi) printf '%s\n' "${KIMI_REASONING_EFFORT:-}" ;;
    qwen) printf '%s\n' "${QWEN_REASONING_EFFORT:-}" ;;
    keenroute) printf '%s\n' "${KEENROUTE_REASONING_EFFORT:-}" ;;
    *) return 2 ;;
  esac
}

check_decoding() {
  case "$1" in greedy|pass8) ;; *) echo "decoding must be greedy or pass8" >&2; return 2 ;; esac
}

verify_setup() {
  "$PY" "$VERIFY"
  "$PY" -m py_compile "$SCRIPT" "$AUDITOR" "$VERIFY"
  bash -n "$0"
}

dry_run() {
  local model="$1" decoding="$2" slug provider url thinking effort adapter_json
  check_decoding "$decoding"
  slug="$(model_slug "$model")"
  provider="$(provider_for_model "$model")"
  url="$(base_url_for_provider "$provider")"
  thinking="$(thinking_mode_for_provider "$provider" "$model")"
  effort="$(reasoning_effort_for_provider "$provider")"
  adapter_json="{\"api_provider\":\"$provider\",\"thinking_mode\":\"$thinking\",\"reasoning_effort\":\"$effort\",\"adapter\":\"official-openai-compatible\"}"
  SFT_API_PROVIDER="$provider" SFT_THINKING_MODE="$thinking" SFT_REASONING_EFFORT="$effort" QWEN_API_KEY="DRY_RUN_ONLY" \
    "$PY" "$SCRIPT" generate \
      --input "$INPUT" \
      --db-root "$DB_ROOT" \
      --schema-cache "$SCHEMA_CACHE" \
      --teacher-prompt "$PROMPT" \
      --output-dir "$MODEL_ROOT/$slug/$decoding" \
      --model-url "$url" \
      --model "$model" \
      --decoding "$decoding" \
      --workers "${SFT_API_WORKERS:-8}" \
      --max-tokens "${SFT_MAX_TOKENS:-8192}" \
      --timeout "${SFT_TIMEOUT:-600}" \
      --max-turns 7 \
      --max-prompt-tokens 16000 \
      --seed 0 \
      --chat-template-kwargs "$adapter_json" \
      --dry-run
}

preflight_model() {
  local model="$1" decoding="$2" provider="$3" url="$4" api_key="$5" thinking="$6" effort="$7"
  local temperature=0
  [[ "$decoding" == pass8 ]] && temperature=1
  if [[ "$provider" == kimi && "$model" == kimi-k2.6 && "$thinking" == disabled ]]; then
    temperature=0.6
  fi
  PREFLIGHT_API_KEY="$api_key" SFT_API_PROVIDER="$provider" \
    SFT_THINKING_MODE="$thinking" SFT_REASONING_EFFORT="$effort" \
    "$PY" - "$ROOT" "$url" "$model" "$temperature" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[1])
import build_sft_traces as B

text = B.request_completion(
    messages=[
        {"role": "system", "content": "You are an API connectivity check."},
        {"role": "user", "content": "Reply with OK."},
    ],
    base_url=sys.argv[2],
    api_key=os.environ["PREFLIGHT_API_KEY"],
    model=sys.argv[3],
    temperature=float(sys.argv[4]),
    max_tokens=32,
    timeout=120,
    seed=0,
    chat_template_kwargs=None,
)
print({"preflight": "ok", "response_chars": len(text)})
PY
}

preflight_job() {
  local model="$1" decoding="$2" provider url api_key thinking effort
  check_decoding "$decoding"
  provider="$(provider_for_model "$model")"
  url="$(base_url_for_provider "$provider")"
  api_key="$(key_for_provider "$provider")"
  thinking="$(thinking_mode_for_provider "$provider" "$model")"
  effort="$(reasoning_effort_for_provider "$provider")"
  echo "preflight model=$model provider=$provider decoding=$decoding url=$url"
  preflight_model "$model" "$decoding" "$provider" "$url" "$api_key" "$thinking" "$effort"
}

check_api() {
  local model="$1" provider url api_key
  provider="$(provider_for_model "$model")"
  url="$(base_url_for_provider "$provider")"
  api_key="$(key_for_provider "$provider")"
  CHECK_API_KEY="$api_key" "$PY" - "$url" "$model" "$provider" <<'PY'
import os, sys
from openai import OpenAI
url, model, provider = sys.argv[1:]
client = OpenAI(api_key=os.environ["CHECK_API_KEY"], base_url=url, timeout=60, max_retries=2)
models = [item.id for item in client.models.list().data]
print({"provider": provider, "base_url": url, "requested_model": model,
       "model_listed": model in models, "available_model_count": len(models)})
if model not in models:
    print("requested model was not returned by GET /models; inspect the provider console/model name")
PY
}

start_job() {
  local model="$1" decoding="$2" slug provider url api_key thinking effort adapter_json workers out log pid_file
  check_decoding "$decoding"
  "$PY" "$VERIFY" >/dev/null
  slug="$(model_slug "$model")"
  provider="$(provider_for_model "$model")"
  url="$(base_url_for_provider "$provider")"
  api_key="$(key_for_provider "$provider")"
  thinking="$(thinking_mode_for_provider "$provider" "$model")"
  effort="$(reasoning_effort_for_provider "$provider")"
  adapter_json="{\"api_provider\":\"$provider\",\"thinking_mode\":\"$thinking\",\"reasoning_effort\":\"$effort\",\"adapter\":\"official-openai-compatible\"}"
  workers="${SFT_API_WORKERS:-8}"
  out="$MODEL_ROOT/$slug/$decoding"
  log="$LOG_ROOT/${slug}_${decoding}.log"
  pid_file="$LOG_ROOT/${slug}_${decoding}.pid"
  mkdir -p "$out" "$LOG_ROOT"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$model/$decoding is already running with PID $(cat "$pid_file")" >&2
    return 1
  fi

  echo "preflight model=$model provider=$provider decoding=$decoding url=$url"
  preflight_model "$model" "$decoding" "$provider" "$url" "$api_key" "$thinking" "$effort"

  export QWEN_API_KEY="$api_key"
  export SFT_API_PROVIDER="$provider"
  export SFT_THINKING_MODE="$thinking"
  export SFT_REASONING_EFFORT="$effort"
  export SFT_MAX_RETRIES="${SFT_MAX_RETRIES:-4}"
  nohup "$PY" "$SCRIPT" generate \
    --input "$INPUT" \
    --db-root "$DB_ROOT" \
    --schema-cache "$SCHEMA_CACHE" \
    --teacher-prompt "$PROMPT" \
    --output-dir "$out" \
    --model-url "$url" \
    --model "$model" \
    --decoding "$decoding" \
    --workers "$workers" \
    --max-tokens "${SFT_MAX_TOKENS:-8192}" \
    --timeout "${SFT_TIMEOUT:-600}" \
    --max-turns 7 \
    --max-prompt-tokens 16000 \
    --seed 0 \
    --chat-template-kwargs "$adapter_json" \
    >>"$log" 2>&1 &
  echo $! >"$pid_file"
  unset QWEN_API_KEY SFT_API_PROVIDER SFT_THINKING_MODE SFT_REASONING_EFFORT
  echo "started model=$model provider=$provider decoding=$decoding pid=$(cat "$pid_file")"
  echo "url=$url thinking_mode=$thinking reasoning_effort=${effort:-provider-default} workers=$workers"
  echo "log=$log"
  echo "output=$out"
}

status_jobs() {
  mkdir -p "$LOG_ROOT"
  local found=0 pid_file pid state name
  shopt -s nullglob
  for pid_file in "$LOG_ROOT"/*.pid; do
    found=1
    pid="$(cat "$pid_file")"
    name="$(basename "$pid_file" .pid)"
    state=stopped
    kill -0 "$pid" 2>/dev/null && state=running
    echo "$name pid=$pid state=$state log=$LOG_ROOT/$name.log"
  done
  shopt -u nullglob
  [[ "$found" -eq 1 ]] || echo "no jobs have been started"
}

tail_job() {
  local model="$1" decoding="$2" slug log
  check_decoding "$decoding"
  slug="$(model_slug "$model")"
  log="$LOG_ROOT/${slug}_${decoding}.log"
  [[ -f "$log" ]] || { echo "missing log: $log" >&2; return 1; }
  tail -n "${TAIL_LINES:-80}" -f "$log"
}

ensure_not_running() {
  local slug="$1" decoding pid_file
  pid_file="$LOG_ROOT/${slug}_${decoding}.pid"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "cannot merge while $slug/$decoding is running (PID $(cat "$pid_file"))" >&2
    return 1
  fi
}

audit_mode() {
  local slug="$1" decoding="$2" dir input
  dir="$MODEL_ROOT/$slug/$decoding"
  input="$dir/accepted_sft.jsonl"
  [[ -f "$input" ]] || { echo "missing completed output: $input" >&2; return 1; }
  "$PY" "$AUDITOR" \
    --input "$input" \
    --db-root "$DB_ROOT" \
    --audit-out "$dir/strict_audit.jsonl" \
    --validated-out "$dir/strict_validated_sft.jsonl" \
    --summary "$dir/strict_audit.summary.json" \
    --workers "${SFT_AUDIT_WORKERS:-8}"
}

merge_model() {
  local model="$1" slug out_dir pre final decoding input
  local inputs=()
  slug="$(model_slug "$model")"
  ensure_not_running "$slug" greedy
  ensure_not_running "$slug" pass8
  for decoding in greedy pass8; do
    input="$MODEL_ROOT/$slug/$decoding/accepted_sft.jsonl"
    if [[ -f "$input" ]]; then
      audit_mode "$slug" "$decoding"
      inputs+=("$MODEL_ROOT/$slug/$decoding/strict_validated_sft.jsonl")
    fi
  done
  [[ ${#inputs[@]} -gt 0 ]] || {
    echo "no completed greedy or pass8 output found for $model" >&2
    return 1
  }
  out_dir="$MERGED_ROOT/by_model/$slug"
  pre="$out_dir/sft_privileged_${slug}.pre_audit.jsonl"
  final="$out_dir/sft_privileged_${slug}.jsonl"
  mkdir -p "$out_dir"
  "$PY" "$SCRIPT" merge \
    --inputs "${inputs[@]}" \
    --out "$pre" \
    --summary "$out_dir/merge.summary.json"
  "$PY" "$AUDITOR" \
    --input "$pre" \
    --db-root "$DB_ROOT" \
    --audit-out "$out_dir/final_audit.jsonl" \
    --validated-out "$final" \
    --summary "$out_dir/final_audit.summary.json" \
    --workers "${SFT_AUDIT_WORKERS:-8}"
  echo "model merge complete: $final"
}

merge_all() {
  local model_dir slug config model final
  local inputs=()
  shopt -s nullglob
  local dirs=("$MODEL_ROOT"/*)
  shopt -u nullglob
  [[ ${#dirs[@]} -gt 0 ]] || { echo "no model outputs under $MODEL_ROOT" >&2; return 1; }
  for model_dir in "${dirs[@]}"; do
    [[ -d "$model_dir" ]] || continue
    slug="$(basename "$model_dir")"
    config="$model_dir/greedy/config.json"
    [[ -f "$config" ]] || config="$model_dir/pass8/config.json"
    [[ -f "$config" ]] || { echo "missing config under $model_dir" >&2; return 1; }
    model="$("$PY" - "$config" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["model"])
PY
)"
    merge_model "$model"
    final="$MERGED_ROOT/by_model/$slug/sft_privileged_${slug}.jsonl"
    inputs+=("$final")
  done
  [[ ${#inputs[@]} -gt 0 ]] || { echo "no per-model merged files" >&2; return 1; }
  mkdir -p "$MERGED_ROOT/all"
  "$PY" "$SCRIPT" merge \
    --inputs "${inputs[@]}" \
    --out "$MERGED_ROOT/all/sft_privileged_kimi_ds_merged.pre_audit.jsonl" \
    --summary "$MERGED_ROOT/all/merge.summary.json"
  "$PY" "$AUDITOR" \
    --input "$MERGED_ROOT/all/sft_privileged_kimi_ds_merged.pre_audit.jsonl" \
    --db-root "$DB_ROOT" \
    --audit-out "$MERGED_ROOT/all/final_audit.jsonl" \
    --validated-out "$MERGED_ROOT/all/sft_privileged_kimi_ds_merged.jsonl" \
    --summary "$MERGED_ROOT/all/final_audit.summary.json" \
    --workers "${SFT_AUDIT_WORKERS:-8}"
  echo "all-model merge complete: $MERGED_ROOT/all/sft_privileged_kimi_ds_merged.jsonl"
}

compare_outputs() {
  "$PY" - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for summary in sorted((root / "models").glob("*/*/summary.json")):
    data = json.loads(summary.read_text(encoding="utf-8"))
    rows.append({
        "run": str(summary.parent.relative_to(root)),
        "attempts": data.get("attempts"),
        "accepted": data.get("accepted"),
        "samples_with_accepted_trace": data.get("samples_with_accepted_trace"),
        "acceptance_rate": data.get("acceptance_rate"),
        "issues": data.get("issues"),
    })
for summary in sorted((root / "merged").glob("**/*.summary.json")):
    data = json.loads(summary.read_text(encoding="utf-8"))
    rows.append({
        "run": str(summary.relative_to(root)),
        "merged_records": data.get("merged_records"),
        "hard_pass_records": data.get("hard_pass_records"),
        "hard_fail_records": data.get("hard_fail_records"),
        "issue_counts": data.get("issue_counts"),
    })
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
}

case "${1:-}" in
  verify) verify_setup ;;
  check-api) [[ $# -eq 2 ]] || { usage; exit 2; }; check_api "$2" ;;
  preflight) [[ $# -eq 3 ]] || { usage; exit 2; }; preflight_job "$2" "$3" ;;
  dry-run) [[ $# -eq 3 ]] || { usage; exit 2; }; dry_run "$2" "$3" ;;
  start) [[ $# -eq 3 ]] || { usage; exit 2; }; start_job "$2" "$3" ;;
  status) status_jobs ;;
  tail) [[ $# -eq 3 ]] || { usage; exit 2; }; tail_job "$2" "$3" ;;
  merge-model) [[ $# -eq 2 ]] || { usage; exit 2; }; merge_model "$2" ;;
  merge-all) merge_all ;;
  compare) compare_outputs ;;
  *) usage; exit 2 ;;
esac
