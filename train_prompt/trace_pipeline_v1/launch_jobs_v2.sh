#!/usr/bin/env bash
set -euo pipefail
# No-gold easy-band copy of ../sft_easy_teacher_traces/launch_easy_teacher_jobs.sh.
# trace_pipeline_v1 copy of ../sft_easy_nogold_v41/launch_nogold_jobs.sh: the ONLY change is SCRIPT
# (build_traces_v2.py = v4.1 generator + entity repair, turn resampling, telemetry).
# Normally driven by pipeline.py, which adds the preflight/smoke gate, run guard and audit.
# Differences: SCRIPT, AUDITOR (1-6 tool calls), PROMPT (v4.1, no appendix), OUTPUT_ROOT, verify --input-root, OpenRouter provider.

PY="${PY:-/data/keenSQL/.venv/bin/python}"
ROOT="/data/keenSQL/train_prompt/sft_easy_teacher_traces"
NOGOLD_ROOT="/data/keenSQL/train_prompt/sft_easy_nogold_v41"
SCRIPT="/data/keenSQL/train_prompt/trace_pipeline_v1/build_traces_v2.py"
AUDITOR="$NOGOLD_ROOT/audit_nogold_traces.py"
PREPARE="$ROOT/prepare_easy_inputs.py"
VERIFY="$ROOT/verify_setup.py"
EXTRACT="$ROOT/extract_existing_teacher.py"
COMBINE="$ROOT/combine_with_existing_teacher.py"
PROMPT="/data/keenSQL/train_prompt/retoolsql_prompt/v4.1/no_teacher_appendix.txt"

OUTPUT_ROOT="${SFT_EASY_NOGOLD_OUTPUT_ROOT:-/data/keenSQL/dataset/m_schema/sft_data/easy_nogold_v41}"
INPUT_ROOT="$OUTPUT_ROOT/inputs"
CHUNK_ROOT="$INPUT_ROOT/chunks"
MODEL_ROOT="$OUTPUT_ROOT/models"
MERGED_ROOT="$OUTPUT_ROOT/merged"
LOG_ROOT="$OUTPUT_ROOT/logs"
EXISTING_ROOT="$OUTPUT_ROOT/existing_teacher"
# trace_pipeline_v1: overridable for non-training database sets (BIRD dev eval); defaults unchanged.
DB_ROOT="${SFT_DB_ROOT:-/data/keenSQL/dataset/all_dataset/databases}"
SCHEMA_CACHE="${SFT_SCHEMA_CACHE:-/data/keenSQL/dataset/m_schema/cache}"

usage() {
  cat <<'EOF'
usage:
  launch_nogold_jobs.sh prepare [CHUNKS]
  launch_nogold_jobs.sh extract-existing
  launch_nogold_jobs.sh verify
  launch_nogold_jobs.sh check-api MODEL
  launch_nogold_jobs.sh preflight MODEL {greedy|pass8}
  launch_nogold_jobs.sh dry-run MODEL {greedy|pass8} CHUNK
  launch_nogold_jobs.sh start MODEL {greedy|pass8} CHUNK
  launch_nogold_jobs.sh start-all MODEL {greedy|pass8}
  launch_nogold_jobs.sh status
  launch_nogold_jobs.sh tail MODEL {greedy|pass8} CHUNK
  launch_nogold_jobs.sh merge-model MODEL
  launch_nogold_jobs.sh merge-all
  launch_nogold_jobs.sh combine-current {MODEL|all}
  launch_nogold_jobs.sh compare

CHUNK accepts either 0 or chunk_000.

Required key:
  KIMI_API_KEY, DEEPSEEK_API_KEY, QWEN_API_KEY, KEENROUTE_API_KEY, OPENROUTER_API_KEY,
  or generic TEACHER_API_KEY.

Concurrency:
  SFT_API_WORKERS is per chunk (default 2). start-all launches every chunk,
  so 8 chunks x 2 workers gives up to 16 concurrent API episodes.
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
      deepseek|kimi|qwen|keenroute|openrouter) printf '%s\n' "$override"; return 0 ;;
      *) echo "unsupported SFT_PROVIDER_OVERRIDE: $override" >&2; return 2 ;;
    esac
  fi
  lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$lower" in
    */*) echo openrouter ;;  # OpenRouter ids are vendor/model, e.g. openai/gpt-5.6-sol
    *deepseek*) echo deepseek ;;
    *kimi*|*moonshot*) echo kimi ;;
    *qwen*) echo qwen ;;
    gpt-*|*astra*|*claude*|*gemini*) echo keenroute ;;
    *)
      echo "cannot infer provider from '$1'; set SFT_PROVIDER_OVERRIDE" >&2
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
      elif [[ "${KIMI_REGION:-cn}" == global ]]; then
        printf '%s\n' "https://api.moonshot.ai/v1"
      else
        printf '%s\n' "https://api.moonshot.cn/v1"
      fi
      ;;
    qwen) printf '%s\n' "${QWEN_BASE_URL:-https://maas.qianwenaiapi.com/compatible-mode/v1}" ;;
    keenroute) printf '%s\n' "${KEENROUTE_BASE_URL:?set KEENROUTE_BASE_URL to the KeenRoute OpenAI-compatible base URL}" ;;
    openrouter) printf '%s\n' "${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}" ;;
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
    openrouter) key="${OPENROUTER_API_KEY:-${TEACHER_API_KEY:-}}" ;;
    *) return 2 ;;
  esac
  if [[ -z "$key" ]]; then
    echo "missing API key for provider $1" >&2
    return 2
  fi
  printf '%s\n' "$key"
}

thinking_mode_for_provider() {
  local provider="$1" model="${2:-}"
  case "$provider" in
    deepseek) printf '%s\n' "${DEEPSEEK_THINKING_MODE:-disabled}" ;;
    kimi)
      if [[ -n "${KIMI_THINKING_MODE+x}" ]]; then
        printf '%s\n' "$KIMI_THINKING_MODE"
      elif [[ "$model" == "kimi-k2.6" ]]; then
        printf '%s\n' disabled
      else
        printf '%s\n' omit
      fi
      ;;
    qwen) printf '%s\n' "${QWEN_THINKING_MODE:-omit}" ;;
    keenroute) printf '%s\n' "${KEENROUTE_THINKING_MODE:-omit}" ;;
    openrouter) printf '%s\n' "${OPENROUTER_THINKING_MODE:-omit}" ;;
    *) return 2 ;;
  esac
}

reasoning_effort_for_provider() {
  case "$1" in
    deepseek) printf '%s\n' "${DEEPSEEK_REASONING_EFFORT:-}" ;;
    kimi) printf '%s\n' "${KIMI_REASONING_EFFORT:-}" ;;
    qwen) printf '%s\n' "${QWEN_REASONING_EFFORT:-}" ;;
    keenroute) printf '%s\n' "${KEENROUTE_REASONING_EFFORT:-}" ;;
    openrouter) printf '%s\n' "${OPENROUTER_REASONING_EFFORT:-}" ;;
    *) return 2 ;;
  esac
}

check_decoding() {
  case "$1" in
    greedy|pass8) ;;
    *) echo "decoding must be greedy or pass8" >&2; return 2 ;;
  esac
}

chunk_name() {
  local raw="$1"
  if [[ "$raw" =~ ^chunk_[0-9]{3}$ ]]; then
    printf '%s\n' "$raw"
  elif [[ "$raw" =~ ^[0-9]+$ ]]; then
    printf 'chunk_%03d\n' "$((10#$raw))"
  else
    echo "invalid chunk '$raw' (use 0 or chunk_000)" >&2
    return 2
  fi
}

chunk_files() {
  shopt -s nullglob
  local files=("$CHUNK_ROOT"/chunk_*.json)
  shopt -u nullglob
  [[ ${#files[@]} -gt 0 ]] || {
    echo "no chunks found; run: bash $0 prepare" >&2
    return 1
  }
  printf '%s\n' "${files[@]}"
}

prepare_inputs() {
  local chunks="${1:-${SFT_EASY_CHUNKS:-8}}"
  "$PY" "$PREPARE" --chunks "$chunks"
}

verify_setup() {
  "$PY" -m py_compile "$PREPARE" "$SCRIPT" "$AUDITOR" "$VERIFY" "$EXTRACT" "$COMBINE"
  if [[ -n "${SFT_DB_ROOT:-}${SFT_SCHEMA_CACHE:-}" ]]; then
    # verify_setup.py hard-codes the training DB root and schema cache; for another database set
    # (BIRD dev) run the same checks against the configured roots instead.
    "$PY" - "$INPUT_ROOT" "$DB_ROOT" "$SCHEMA_CACHE" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, db_root, schema = map(Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
m = json.loads((root / "manifest.json").read_text())
assert sha(m["all_input"]) == m["all_input_sha256"], "input hash mismatch"
full = json.loads(Path(m["all_input"]).read_text())
ids, dbs = [], set()
for c in m["chunk_records"]:
    assert sha(c["path"]) == c["sha256"], f"chunk hash mismatch {c['path']}"
    rows = json.loads(Path(c["path"]).read_text())
    ids += [str(r["question_id"]) for r in rows]
    dbs |= {r["db_id"] for r in rows}
assert sorted(ids) == sorted(str(r["question_id"]) for r in full) and len(ids) == len(set(ids)), "chunks != input"
bad_db = [d for d in dbs if not (db_root / d / f"{d}.sqlite").is_file()]
bad_schema = [d for d in dbs if not (schema / f"{d}.txt").is_file()]
assert not bad_db and not bad_schema, f"missing databases={bad_db} schemas={bad_schema}"
print(json.dumps({"verify": "ok", "records": len(full), "databases": len(dbs), "db_root": str(db_root), "schema_cache": str(schema)}))
PY
  else
    "$PY" "$VERIFY" --input-root "$INPUT_ROOT"
  fi
  bash -n "$0"
}

extract_existing() {
  "$PY" "$EXTRACT"
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
    print("requested model was not returned by GET /models")
PY
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
import build_easy_teacher_traces as E
text = E.adapter.request_completion(
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

dry_run() {
  local model="$1" decoding="$2" chunk slug provider url thinking effort input out adapter_json
  check_decoding "$decoding"
  chunk="$(chunk_name "$3")"
  input="$CHUNK_ROOT/$chunk.json"
  [[ -f "$input" ]] || { echo "missing chunk: $input" >&2; return 1; }
  slug="$(model_slug "$model")"
  provider="$(provider_for_model "$model")"
  url="$(base_url_for_provider "$provider")"
  thinking="$(thinking_mode_for_provider "$provider" "$model")"
  effort="$(reasoning_effort_for_provider "$provider")"
  out="$MODEL_ROOT/$slug/$decoding/$chunk"
  adapter_json="{\"api_provider\":\"$provider\",\"thinking_mode\":\"$thinking\",\"reasoning_effort\":\"$effort\",\"input_chunk\":\"$chunk\"}"
  SFT_API_PROVIDER="$provider" SFT_THINKING_MODE="$thinking" \
    SFT_REASONING_EFFORT="$effort" QWEN_API_KEY=DRY_RUN_ONLY \
    "$PY" "$SCRIPT" generate \
      --input "$input" --db-root "$DB_ROOT" --schema-cache "$SCHEMA_CACHE" \
      --teacher-prompt "$PROMPT" --output-dir "$out" \
      --model-url "$url" --model "$model" --decoding "$decoding" \
      --workers "${SFT_API_WORKERS:-2}" --max-tokens "${SFT_MAX_TOKENS:-8192}" \
      --timeout "${SFT_TIMEOUT:-600}" --max-turns 7 --max-prompt-tokens "${SFT_MAX_PROMPT_TOKENS:-16000}" \
      --seed 0 --chat-template-kwargs "$adapter_json" --dry-run
}

start_one() {
  local model="$1" decoding="$2" chunk="$3" skip_preflight="${4:-0}"
  local slug provider url api_key thinking effort adapter_json workers input out log_dir log pid_file
  check_decoding "$decoding"
  chunk="$(chunk_name "$chunk")"
  input="$CHUNK_ROOT/$chunk.json"
  [[ -f "$input" ]] || { echo "missing chunk: $input" >&2; return 1; }
  slug="$(model_slug "$model")"
  provider="$(provider_for_model "$model")"
  url="$(base_url_for_provider "$provider")"
  api_key="$(key_for_provider "$provider")"
  thinking="$(thinking_mode_for_provider "$provider" "$model")"
  effort="$(reasoning_effort_for_provider "$provider")"
  adapter_json="{\"api_provider\":\"$provider\",\"thinking_mode\":\"$thinking\",\"reasoning_effort\":\"$effort\",\"input_chunk\":\"$chunk\"}"
  workers="${SFT_API_WORKERS:-2}"
  out="$MODEL_ROOT/$slug/$decoding/$chunk"
  log_dir="$LOG_ROOT/$slug/$decoding"
  log="$log_dir/$chunk.log"
  pid_file="$log_dir/$chunk.pid"
  mkdir -p "$out" "$log_dir"
  if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$model/$decoding/$chunk already running pid=$(cat "$pid_file")"
    return 0
  fi
  if [[ "$skip_preflight" != 1 ]]; then
    preflight_model "$model" "$decoding" "$provider" "$url" "$api_key" "$thinking" "$effort"
  fi
  QWEN_API_KEY="$api_key" SFT_API_PROVIDER="$provider" \
    SFT_THINKING_MODE="$thinking" SFT_REASONING_EFFORT="$effort" \
    SFT_MAX_RETRIES="${SFT_MAX_RETRIES:-4}" \
    nohup "$PY" "$SCRIPT" generate \
      --input "$input" --db-root "$DB_ROOT" --schema-cache "$SCHEMA_CACHE" \
      --teacher-prompt "$PROMPT" --output-dir "$out" \
      --model-url "$url" --model "$model" --decoding "$decoding" \
      --workers "$workers" --max-tokens "${SFT_MAX_TOKENS:-8192}" \
      --timeout "${SFT_TIMEOUT:-600}" --max-turns 7 --max-prompt-tokens "${SFT_MAX_PROMPT_TOKENS:-16000}" \
      --seed 0 --chat-template-kwargs "$adapter_json" \
      >>"$log" 2>&1 < /dev/null &
  echo $! >"$pid_file"
  echo "started model=$model decoding=$decoding chunk=$chunk pid=$(cat "$pid_file") workers=$workers"
  echo "log=$log"
  echo "output=$out"
}

start_all() {
  local model="$1" decoding="$2" provider url api_key thinking effort file chunk
  verify_setup >/dev/null
  check_decoding "$decoding"
  provider="$(provider_for_model "$model")"
  url="$(base_url_for_provider "$provider")"
  api_key="$(key_for_provider "$provider")"
  thinking="$(thinking_mode_for_provider "$provider" "$model")"
  effort="$(reasoning_effort_for_provider "$provider")"
  echo "one preflight before parallel chunk launch: model=$model decoding=$decoding"
  preflight_model "$model" "$decoding" "$provider" "$url" "$api_key" "$thinking" "$effort"
  while IFS= read -r file; do
    chunk="$(basename "$file" .json)"
    start_one "$model" "$decoding" "$chunk" 1
  done < <(chunk_files)
  echo "all chunks launched; per_chunk_workers=${SFT_API_WORKERS:-2}"
}

status_jobs() {
  local found=0 pid_file pid state rel slug decoding chunk output raw completed accepted
  mkdir -p "$LOG_ROOT"
  while IFS= read -r pid_file; do
    found=1
    pid="$(cat "$pid_file")"
    rel="${pid_file#"$LOG_ROOT"/}"
    slug="${rel%%/*}"
    rel="${rel#*/}"
    decoding="${rel%%/*}"
    chunk="$(basename "$pid_file" .pid)"
    output="$MODEL_ROOT/$slug/$decoding/$chunk"
    state=stopped
    kill -0 "$pid" 2>/dev/null && state=running
    [[ -f "$output/summary.json" ]] && state=completed
    raw=0
    [[ -f "$output/raw_attempts.jsonl" ]] && raw="$(wc -l < "$output/raw_attempts.jsonl")"
    completed="-"
    accepted="-"
    if [[ -f "$output/summary.json" ]]; then
      read -r completed accepted < <("$PY" - "$output/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
print(d.get("attempts", "-"), d.get("accepted", "-"))
PY
)
    fi
    echo "$slug/$decoding/$chunk pid=$pid state=$state raw_attempts=$raw attempts=$completed accepted=$accepted"
  done < <(find "$LOG_ROOT" -type f -name 'chunk_*.pid' | sort)
  [[ "$found" -eq 1 ]] || echo "no easy-teacher jobs have been started"
}

tail_job() {
  local model="$1" decoding="$2" chunk slug log
  check_decoding "$decoding"
  chunk="$(chunk_name "$3")"
  slug="$(model_slug "$model")"
  log="$LOG_ROOT/$slug/$decoding/$chunk.log"
  [[ -f "$log" ]] || { echo "missing log: $log" >&2; return 1; }
  tail -n "${TAIL_LINES:-80}" -f "$log"
}

ensure_model_not_running() {
  local slug="$1" pid_file pid
  while IFS= read -r pid_file; do
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "cannot merge while running: $pid_file pid=$pid" >&2
      return 1
    fi
  done < <(find "$LOG_ROOT/$slug" -type f -name '*.pid' 2>/dev/null | sort)
}

audit_chunk() {
  local dir="$1" input
  input="$dir/accepted_sft.jsonl"
  [[ -f "$input" ]] || { echo "missing completed output: $input" >&2; return 1; }
  "$PY" "$AUDITOR" \
    --input "$input" --db-root "$DB_ROOT" \
    --audit-out "$dir/strict_audit.jsonl" \
    --validated-out "$dir/strict_validated_sft.jsonl" \
    --summary "$dir/strict_audit.summary.json" \
    --workers "${SFT_AUDIT_WORKERS:-8}"
}

merge_model() {
  local model="$1" slug mode file chunk dir found missing out_dir pre final
  local require_both="${SFT_REQUIRE_BOTH_MODES:-1}"
  local inputs=()
  slug="$(model_slug "$model")"
  ensure_model_not_running "$slug"
  for mode in greedy pass8; do
    found=0
    missing=0
    while IFS= read -r file; do
      chunk="$(basename "$file" .json)"
      dir="$MODEL_ROOT/$slug/$mode/$chunk"
      if [[ -f "$dir/accepted_sft.jsonl" ]]; then
        found=$((found + 1))
      else
        missing=$((missing + 1))
      fi
    done < <(chunk_files)
    if (( found == 0 )); then
      if [[ "$require_both" == 1 ]]; then
        echo "missing all $mode chunks for $model" >&2
        return 1
      fi
      continue
    fi
    if (( missing > 0 )); then
      echo "$model/$mode is incomplete: found=$found missing=$missing" >&2
      return 1
    fi
    while IFS= read -r file; do
      chunk="$(basename "$file" .json)"
      dir="$MODEL_ROOT/$slug/$mode/$chunk"
      audit_chunk "$dir"
      inputs+=("$dir/strict_validated_sft.jsonl")
    done < <(chunk_files)
  done
  [[ ${#inputs[@]} -gt 0 ]] || { echo "no completed inputs for $model" >&2; return 1; }
  out_dir="$MERGED_ROOT/by_model/$slug"
  pre="$out_dir/sft_easy_teacher_${slug}.pre_audit.jsonl"
  final="$out_dir/sft_easy_teacher_${slug}.jsonl"
  mkdir -p "$out_dir"
  "$PY" "$SCRIPT" merge --inputs "${inputs[@]}" --out "$pre" \
    --summary "$out_dir/merge.summary.json"
  "$PY" "$AUDITOR" --input "$pre" --db-root "$DB_ROOT" \
    --audit-out "$out_dir/final_audit.jsonl" --validated-out "$final" \
    --summary "$out_dir/final_audit.summary.json" \
    --workers "${SFT_AUDIT_WORKERS:-8}"
  echo "model merge complete: $final"
}

merge_all() {
  local model_dir slug config model final out_dir pre out
  local inputs=()
  shopt -s nullglob
  local dirs=("$MODEL_ROOT"/*)
  shopt -u nullglob
  [[ ${#dirs[@]} -gt 0 ]] || { echo "no model outputs under $MODEL_ROOT" >&2; return 1; }
  for model_dir in "${dirs[@]}"; do
    [[ -d "$model_dir" ]] || continue
    slug="$(basename "$model_dir")"
    config="$(find "$model_dir" -type f -name config.json | sort | head -1)"
    [[ -n "$config" ]] || { echo "missing config under $model_dir" >&2; return 1; }
    model="$("$PY" - "$config" <<'PY'
import json, sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["model"])
PY
)"
    merge_model "$model"
    final="$MERGED_ROOT/by_model/$slug/sft_easy_teacher_${slug}.jsonl"
    inputs+=("$final")
  done
  out_dir="$MERGED_ROOT/all"
  pre="$out_dir/sft_easy_teacher_all_models.pre_audit.jsonl"
  out="$out_dir/sft_easy_teacher_all_models.jsonl"
  mkdir -p "$out_dir"
  "$PY" "$SCRIPT" merge --inputs "${inputs[@]}" --out "$pre" \
    --summary "$out_dir/merge.summary.json"
  "$PY" "$AUDITOR" --input "$pre" --db-root "$DB_ROOT" \
    --audit-out "$out_dir/final_audit.jsonl" --validated-out "$out" \
    --summary "$out_dir/final_audit.summary.json" \
    --workers "${SFT_AUDIT_WORKERS:-8}"
  echo "all-model merge complete: $out"
}

combine_current() {
  local target="$1" slug easy out_dir out summary
  local existing="$EXISTING_ROOT/current_teacher_1781.jsonl"
  [[ -f "$existing" ]] || { echo "missing $existing; run extract-existing" >&2; return 1; }
  if [[ "$target" == all ]]; then
    easy="$MERGED_ROOT/all/sft_easy_teacher_all_models.jsonl"
    out_dir="$MERGED_ROOT/final/all"
  else
    slug="$(model_slug "$target")"
    easy="$MERGED_ROOT/by_model/$slug/sft_easy_teacher_${slug}.jsonl"
    out_dir="$MERGED_ROOT/final/$slug"
  fi
  [[ -f "$easy" ]] || { echo "missing easy merged result: $easy" >&2; return 1; }
  out="$out_dir/sft_current_teacher_plus_regenerated_easy.jsonl"
  summary="$out_dir/summary.json"
  "$PY" "$COMBINE" --existing "$existing" --easy "$easy" --out "$out" --summary "$summary"
  echo "combined current teacher + regenerated easy: $out"
}

compare_outputs() {
  "$PY" - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for path in sorted((root/"models").glob("*/*/chunk_*/summary.json")):
    d=json.loads(path.read_text(encoding="utf-8"))
    rows.append({"run":str(path.parent.relative_to(root)),"attempts":d.get("attempts"),
                 "accepted":d.get("accepted"),"samples":d.get("samples_with_accepted_trace"),
                 "issues":d.get("issues")})
for path in sorted((root/"merged").glob("**/*.summary.json")):
    d=json.loads(path.read_text(encoding="utf-8"))
    rows.append({"run":str(path.relative_to(root)),"merged":d.get("merged_records"),
                 "hard_pass":d.get("hard_pass_records"),"hard_fail":d.get("hard_fail_records"),
                 "issues":d.get("issue_counts")})
print(json.dumps(rows,ensure_ascii=False,indent=2))
PY
}

case "${1:-}" in
  prepare|extract-existing|combine-current) echo "$1 belongs to the privileged easy-teacher pipeline; not used by the no-gold run" >&2; exit 2 ;;
  verify) [[ $# -eq 1 ]] || { usage; exit 2; }; verify_setup ;;
  check-api) [[ $# -eq 2 ]] || { usage; exit 2; }; check_api "$2" ;;
  preflight) [[ $# -eq 3 ]] || { usage; exit 2; }; preflight_job "$2" "$3" ;;
  dry-run) [[ $# -eq 4 ]] || { usage; exit 2; }; dry_run "$2" "$3" "$4" ;;
  start) [[ $# -eq 4 ]] || { usage; exit 2; }; verify_setup >/dev/null; start_one "$2" "$3" "$4" ;;
  start-all) [[ $# -eq 3 ]] || { usage; exit 2; }; start_all "$2" "$3" ;;
  status) [[ $# -eq 1 ]] || { usage; exit 2; }; status_jobs ;;
  tail) [[ $# -eq 4 ]] || { usage; exit 2; }; tail_job "$2" "$3" "$4" ;;
  merge-model) [[ $# -eq 2 ]] || { usage; exit 2; }; merge_model "$2" ;;
  merge-all) [[ $# -eq 1 ]] || { usage; exit 2; }; merge_all ;;
  compare) [[ $# -eq 1 ]] || { usage; exit 2; }; compare_outputs ;;
  *) usage; exit 2 ;;
esac
