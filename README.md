# keenSQL teacher-trace generation (BIRD Text-to-SQL)

Code for generating and auditing teacher trajectories (SFT traces) for a ReToolSQL-style agent: the student calls read-only SQLite tools turn by turn and submits one final SQL. Teachers (GPT-5.6 Sol, Claude Opus 5.5, Kimi, DeepSeek, ...) are called over OpenAI-compatible APIs (OpenRouter by default).

Copied from `/data/keenSQL/` with the same relative layout, because the scripts use absolute paths under `/data/keenSQL`. Datasets, databases, M-Schema caches and generated traces are **not** included. API keys are read only from environment variables (`OPENROUTER_API_KEY`, `KIMI_API_KEY`, `DEEPSEEK_API_KEY`, ...). The internal KeenRoute default address was removed: set `KEENROUTE_BASE_URL` to use the `keenroute` provider (scripts stop with an error if it is unset).

## Reading order

1. **Overview**: `train_prompt/trace_pipeline_v1/README.md`, which covers the gate → generate → audit → adjudicate → resample → release loop.
2. **Prompt**: `train_prompt/retoolsql_prompt/v4.1/` (`system_prompt.txt`, `fewshot.json`, `user_template.txt`) and `train_prompt/retoolsql_prompt/prompt_builder.py` (`build_messages`).
3. **Tools and grading**:
   - `database_tools/sqlite_tools.py`: `sqlite_query`, `sqlite_peek`, `bm25_search_sqlite`
   - `train_prompt/test_retoolsql_prompt.py`: turn parser and protocol checks
   - `train_prompt/sql_grading.py`: EX
4. **Core loop**: `train_prompt/sft_privileged_traces/build_sft_traces.py`. Read in this order: `build_prompts`, `request_completion`, `run_episode` (the multi-turn loop), `grade_final`, `acceptance_audit`, `run_attempt`, `generate`.
5. **Wrapper layers**: each layer imports the previous one and patches a few functions.
   1. `train_prompt/sft_privileged_traces_kimi_ds/build_sft_traces.py`: sha256 lock and OpenAI-compatible client
   2. `train_prompt/sft_easy_teacher_traces/build_easy_teacher_traces.py`: easy-tier records
   3. `train_prompt/sft_easy_nogold_v41/build_nogold_traces.py`: no gold in the prompt, v4.1 prompt, stop and clip at `</tool_call>`
   4. `train_prompt/trace_pipeline_v1/build_traces_v2.py`: entity repair, turn-level resampling, telemetry, Anthropic prompt caching, and OpenRouter provider pinning (`SFT_OPENROUTER_PROVIDERS`)
6. **Orchestration** (all in `train_prompt/trace_pipeline_v1/`):
   - `launch_jobs_v2.sh`: provider routing and chunked launch
   - `pipeline.py`: stage entry point
   - `preflight.py`, `scan_traces.py`, `run_guard.py`
   - `audit_l3.py`, `audit_l3b.py`: two-judge audit and adjudication

## Example: Claude Opus 5.5 run

```bash
export OPENROUTER_API_KEY=...            # environment only
export SFT_OPENROUTER_PROVIDERS=anthropic,google-vertex   # Bedrock / Claude-on-AWS refuse the v4.1 prompt
PL=".venv/bin/python train_prompt/trace_pipeline_v1/pipeline.py"
$PL gate     --root $R --teacher anthropic/claude-opus-5.5 --budget 70 --floor 5
$PL smoke    --root $R --teacher anthropic/claude-opus-5.5 --n 10
$PL generate --root $R --teacher anthropic/claude-opus-5.5 --workers 2 --floor 5 --wait
```

For reference, Opus 5.5 single greedy on BIRD dev (1534 questions, official evaluator) scores 81.94% EX. Traces generated from BIRD dev are for evaluation only and must never be used for training.
