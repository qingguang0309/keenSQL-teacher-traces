# 非现有教师样本的教师轨迹再生成

## 范围

- 原始全集共 11,358 条，已经具有教师轨迹的 1,781 条单独冻结保存。
- 当前交付的教师轨迹生成输入共 9,119 条：当前 easy 7,508 条，加上此前未进入该
  生成池的 1,611 条。
- 输入固定分为 8 个 chunk；重跑会根据各块的 `raw_attempts.jsonl` 跳过已经完成的
  样本，只请求尚未完成的 attempt。
- 学生提示与教师提示全部来自 `retoolsql_prompt/v4`：
  `system_prompt.txt`、`fewshot.json`、`user_template.txt`、
  `sft_teacher_prompt.txt`。
- greedy 每题 1 条完整轨迹，temperature=0；pass@8 每题 8 条独立完整轨迹，
  temperature=1（厂商固定温度的模型按其接口要求处理）。

## 目录

- 脚本：`/data/keenSQL/train_prompt/sft_easy_teacher_traces`
- 数据：`/data/keenSQL/dataset/sft_data/easy_teacher_traces`
- 切块：`inputs/chunks/chunk_000.json` 等
- 模型运行：`models/<模型>/<greedy|pass8>/<chunk>/`
- 日志：`logs/<模型>/<greedy|pass8>/<chunk>.log`
- 单模型合并：`merged/by_model/<模型>/sft_easy_teacher_<模型>.jsonl`
- 多模型合并：`merged/all/sft_easy_teacher_all_models.jsonl`
- 当前已有教师轨迹：`existing_teacher/current_teacher_1781.jsonl`
- 与当前已有教师轨迹合并：`merged/final/<模型或all>/`
- 生成阶段通过/拒绝合并：`merged/generation_stage/qwen3.6-plus/greedy/`

## 运行

```bash
cd /data/keenSQL
source .venv/bin/activate
cd /data/keenSQL/train_prompt/sft_easy_teacher_traces

# 已默认准备为 8 块；重新核验输入和资产
bash launch_easy_teacher_jobs.sh verify
bash launch_easy_teacher_jobs.sh extract-existing

# 按教师模型导入对应 Key
export QWEN_API_KEY='sk-...'
# export KIMI_API_KEY='sk-...'
# export DEEPSEEK_API_KEY='sk-...'
# export KEENROUTE_API_KEY='sk-...'

bash launch_easy_teacher_jobs.sh check-api qwen3.6-plus
bash launch_easy_teacher_jobs.sh preflight qwen3.6-plus greedy
bash launch_easy_teacher_jobs.sh preflight qwen3.6-plus pass8

# 每个块 2 个 API worker，8 块同时后台运行，总并发最多约 16。
export SFT_API_WORKERS=2
bash launch_easy_teacher_jobs.sh start-all qwen3.6-plus greedy
bash launch_easy_teacher_jobs.sh start-all qwen3.6-plus pass8

bash launch_easy_teacher_jobs.sh status
bash launch_easy_teacher_jobs.sh tail qwen3.6-plus greedy 0
bash launch_easy_teacher_jobs.sh tail qwen3.6-plus pass8 0

# 两种解码和全部块结束后执行
bash launch_easy_teacher_jobs.sh merge-model qwen3.6-plus
bash launch_easy_teacher_jobs.sh compare

# 将该模型产生的教师轨迹与当前已有的 1,781 条教师轨迹取并集
bash launch_easy_teacher_jobs.sh combine-current qwen3.6-plus
```

`start-all` 对模型和解码参数只预检一次，然后用同一 Key 并行启动所有块。每块有
独立 PID、日志、原始尝试和断点状态；同一命令重跑会从各块的
`raw_attempts.jsonl` 继续，不会重做已完成 attempt。关闭终端不影响 nohup 任务。

如果模型不支持 temperature=0，可只跑 pass@8，合并时显式允许单一解码：

```bash
SFT_REQUIRE_BOTH_MODES=0 bash launch_easy_teacher_jobs.sh merge-model kimi-k3
```

## 校验与选择

生成时和合并前后均使用第一版严格规则：独立 EX=True；2～6 次工具调用；最终
SQL 必须被 `sqlite_query` 原样执行；`Verified call` 指向该调用；observation 可
从数据库重放；XML 标签完整；无 `<think>`、特权 SQL 提及和不可见 Gold 字符串。

同一题有多个合格候选时，优先无重复调用、2～4 次有效调用、轮次更少且轨迹更简洁
的一条；greedy 只作为同质量候选的后置 tie-breaker。每个原始 question ID 最终最多
保留一条轨迹。

## 主要产物

每个块：

- `raw_attempts.jsonl`：全部尝试、EX 和失败原因，也是断点依据。
- `accepted_sft.jsonl`：通过生成时硬门槛的候选。
- `rejected_attempts.jsonl`：被拒候选及原因。
- `summary.json`：尝试数、通过数和问题分布。
- `strict_audit.jsonl`：数据库重放后的逐条审核结果。
- `strict_validated_sft.jsonl`：可进入合并的严格合格轨迹。

单模型结果会把所有块的 greedy 与 pass@8 候选先严格审核，再按题选择一条。若运行
多个教师模型，`merge-all` 会在各单模型结果之间再次按质量选择：

```bash
bash launch_easy_teacher_jobs.sh merge-all
bash launch_easy_teacher_jobs.sh combine-current all
```

`combine-current` 按 `question_id` 校验唯一性并要求两个集合没有交叉；不会静默覆盖。
当前生成池为 9,119 条实际落盘样本；与已有 1,781 条教师轨迹的理论最大并集为
10,900 条。
