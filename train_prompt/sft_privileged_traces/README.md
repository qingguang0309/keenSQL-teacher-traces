# Privileged reasoning traces 构建

本目录为 SFT 阶段 hard 样本生成带 gold 引导的教师轨迹。目标不是教模型背 gold
SQL，而是在**干净的 v4 部署提示**下蒸馏一条可复现的多轮工具轨迹：学生输入永远
不含 gold 或教师块，只有 assistant 轮进入 loss。

论文对齐（ReToolSQL，§3.10）：SFT 的 hard 样本由教师带 gold 参照生成，经执行
校验后只保留正确轨迹；easy 样本使用策略自身无特权成功轨迹。RLVR 的 rollout 输入
与 SFT 相同，gold 只进入 reward。BIRD 测试输入没有任何特权信息。

## 生成逻辑

输入为 pass@16 全错的 hard 样本
`dataset/sft_data/data/all_dataset_qwen35_9b_pass16_full_pcache_bt16384_wrong.json`。
生成时沿用 v4 的 `system_prompt.txt`、`fewshot.json`、`user_template.txt`，并在
当前问题后追加 `v4/sft_teacher_prompt.txt`（唯一教师提示，gold-only）：

- 教师首轮必须命名 2–3 个决定性 schema fact 并给出自己的草稿理由；
- **至少 2 次工具调用，通常 2–4 次，复杂样本最多 6 次**：第一次是决策性验证
  （bm25 / peek / 窄 SQL），最后必须是最终 SQL 的精确执行；每个额外调用都必须
  解决 schema 或上一条 observation 暴露的不同问题；
- 禁止 `<think>`，禁止复述或暗示 gold / reference / privileged / teacher；
- 最多 7 个 assistant 轮，即最多 6 个工具调用轮加 1 个最终答案轮。

四种运行方式：

| 模型 | 解码 | 每题轨迹数 | temperature |
|---|---|---:|---:|
| Qwen3.5-9B（本地 vLLM） | greedy | 1 | 0 |
| Qwen3.5-9B（本地 vLLM） | pass@8 | 8 | 1 |
| qwen3.6-plus（OpenAI 兼容 API） | greedy | 1 | 0 |
| qwen3.6-plus（OpenAI 兼容 API） | pass@8 | 8 | 1 |

pass@8 完整生成 8 条，不因较早命中而停止。每条最终 SQL 按样本自己的
`grading_method` 做 EX；只保留：

- EX 正确；
- 协议完整、所有标签闭合；
- 最终 SQL 由原样 `sqlite_query` 执行过，且 `Verified call: k` 精确指向它；
- 工具调用 ≥2；
- 工具调用 ≤6，assistant 轮数 ≤7；
- assistant 内容不含 `<think>`、gold/reference/privileged/teacher 字样；
- 推理中未出现未在问题/evidence/schema/observation 中出现过的 gold 字符串字面量。

写入 `accepted_sft.jsonl` 的 `messages` 使用原始无特权 v4 提示，再拼接生成的
assistant/observation 轨迹；gold SQL 教师块不会进入训练输入。记录同时带
`prompt_messages`、`prompt_hash`、`teacher_prompt_sha256`、`data_variant`、
`teacher_model`、`decoding`，供装配和训练前检查使用。

## 后台启动

```bash
cd /data/keenSQL/train_prompt/sft_privileged_traces

# 本地 Qwen3.5-9B；两种模式建议依次运行
bash launch_sft_jobs.sh qwen35-greedy
bash launch_sft_jobs.sh qwen35-pass8

# qwen3.6-plus
export QWEN_BASE_URL='你的 OpenAI 兼容 /v1 地址'
export QWEN_API_KEY='你的 API Key'
bash launch_sft_jobs.sh qwen36plus-greedy
bash launch_sft_jobs.sh qwen36plus-pass8

bash launch_sft_jobs.sh status
bash launch_sft_jobs.sh compare
```

本地默认 `workers=32`，API 默认 `workers=8`；可分别通过 `SFT_LOCAL_WORKERS`、
`SFT_API_WORKERS` 调整。相同模型的 greedy 与 pass@8 建议顺序执行，避免争抢同一
服务吞吐。外部 API 若拒绝 `extra_body.chat_template_kwargs`，客户端会自动去掉
该参数重试。

## 合并

按教师分组合并各解码模式，再按完整 messages 去重：

```bash
bash launch_sft_jobs.sh merge-qwen35
bash launch_sft_jobs.sh merge-qwen36plus
bash launch_sft_jobs.sh merge          # 全部教师合并
```

输出：

- `dataset/sft_data/privileged_traces/sft_privileged_qwen35.jsonl`
- `dataset/sft_data/privileged_traces/sft_privileged_qwen36plus.jsonl`
- `dataset/sft_data/privileged_traces/sft_privileged_merged.jsonl`

## 文件

每个模式输出到 `dataset/sft_data/privileged_traces/<mode>/`：

- `raw_attempts.jsonl`：全部轨迹、EX 与筛选审计，支持断点续跑。
- `accepted_sft.jsonl`：可训练的 SFT messages。
- `rejected_attempts.jsonl`：未通过原因。
- `config.json`：模型、解码、提示词哈希和输入哈希。
- `summary.json`：通过率、EX 命中数及拒绝原因统计。

PID 和日志位于 `/data/keenSQL/logs/data_screen/`（旧路径
`dataset/sft_data/logs/` 现在是指向它的符号链接），文件名前缀为
`sft_privileged_<mode>`。

## 与 SFT 装配的衔接

生成完成后：

```bash
python sft_train/assemble_sft_data.py \
  --easy dataset/sft_data/easy/sft_easy.jsonl \
  --hard dataset/sft_data/privileged_traces/sft_privileged_qwen36plus.jsonl \
  --prefer qwen3.6-plus:greedy,qwen3.6-plus \
  --output dataset/sft_data/assembled/sft_mixed.jsonl
```

默认 `hard_first`：同一道题若既有 easy 成功轨迹又有 hard 特权轨迹，保留 hard。
可用 `--policy easy_first|keep_both` 做消融。完整流程见
[`sft_train/SFT_PIPELINE.md`](../../sft_train/SFT_PIPELINE.md)。
