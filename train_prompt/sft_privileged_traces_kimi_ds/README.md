# Kimi / DeepSeek / Qwen / KeenRoute 第一版特权教师轨迹

## 固定范围

- 输入：`/data/keenSQL/dataset/sft_data/privileged_traces/other_teacher_input/remaining_hard_1382.json`
- 样本数：1,382。
- 学生提示：`retoolsql_prompt/v4` 的 `system_prompt.txt`、`fewshot.json`、`user_template.txt`。
- 教师提示：本目录冻结的 `sft_teacher_prompt_v1.txt`，SHA256 为
  `431146a9ac3e73602ee1c0c69b79e92bca332d5b5aecd529d05e40abb9f68a67`。
- 生成与校验复用第一版脚本，并校验上游脚本哈希，避免静默漂移。

greedy 每题生成 1 条、temperature=0；pass@8 每题完整生成 8 条、temperature=1。
Kimi/DeepSeek 只替换教师模型，不改变第一版提示词及保留规则。

## API 配置

脚本按模型名自动选择官方 OpenAI 兼容地址：

- 模型名含 `kimi` 或 `moonshot`：国内默认 `https://api.moonshot.cn/v1`。
- 模型名含 `deepseek`：默认 `https://api.deepseek.com`。
- 模型名含 `qwen`：默认使用
  `https://maas.qianwenaiapi.com/compatible-mode/v1`，也可用 `QWEN_BASE_URL` 覆盖。
- 模型名为 `gpt-*`，或含 `astra`、`claude`、`gemini`：默认使用 KeenRoute 的
  OpenAI 兼容地址 `$KEENROUTE_BASE_URL`（需自行设置）。其他中转模型可设置
  `SFT_PROVIDER_OVERRIDE=keenroute`，地址可用 `KEENROUTE_BASE_URL` 覆盖。

Kimi 国际平台可设置 `export KIMI_REGION=global`，地址会变为
`https://api.moonshot.ai/v1`。也可用 `KIMI_BASE_URL` 或
`DEEPSEEK_BASE_URL` 显式覆盖。

API Key 只从环境变量读取，不写入配置和日志：

```bash
export KIMI_API_KEY='sk-...'
export DEEPSEEK_API_KEY='sk-...'
export QWEN_API_KEY='sk-...'
export KEENROUTE_API_KEY='sk-...'
```

默认 API 并发为 8，可按账户限额调整：

```bash
export SFT_API_WORKERS=4
```

DeepSeek 默认显式关闭隐藏 thinking，使 temperature=0/1 分别保持 greedy/pass@8
含义。Kimi 默认不发送 thinking 开关，以兼容不同模型系列；必要时可设置
`KIMI_THINKING_MODE=enabled` 或 `disabled`。

Qwen 默认不注入厂商特定 thinking 参数，与第一次审核调用方式一致；响应只保存
`message.content`，教师提示仍禁止输出 `<think>`。如确有需要，可显式设置
`QWEN_THINKING_MODE=enabled` 或 `disabled` 并先运行预检。

KeenRoute 默认采用 OpenAI Chat Completions 格式且不注入厂商特定 thinking 参数。
若中转模型支持推理强度，可设置 `KEENROUTE_REASONING_EFFORT`；若只支持 Anthropic
Messages 而不支持 OpenAI Chat Completions，现有 `preflight` 会在正式启动前失败。

启用 thinking 时可显式设置推理强度，例如：

```bash
export KIMI_REASONING_EFFORT=high
export DEEPSEEK_REASONING_EFFORT=high
export QWEN_REASONING_EFFORT=high
export KEENROUTE_REASONING_EFFORT=high
```

启动前会执行一次最多 32 token 的真实 API 预检，以验证 Key、URL、模型权限、额度
和请求参数。正式运行中，SDK 重试耗尽后出现任何 API/适配器异常都会触发熔断并
终止任务；失败 attempt 不会写成已完成，修复后重新执行同一 `start` 命令即可断点续跑。
也可以先用 `preflight MODEL MODE` 单独测试相应温度等参数，不启动后台任务。

## 运行

```bash
cd /data/keenSQL/train_prompt/sft_privileged_traces_kimi_ds

# 不调用 API：检查输入、数据库、M-Schema、提示词和脚本
bash launch_sft_jobs.sh verify
bash launch_sft_jobs.sh dry-run kimi-k3 pass8
bash launch_sft_jobs.sh dry-run deepseek-v4-pro pass8

# 只调用 GET /models，验证 Key、区域、地址和模型名
bash launch_sft_jobs.sh check-api kimi-k3
bash launch_sft_jobs.sh check-api deepseek-v4-pro
bash launch_sft_jobs.sh check-api qwen3.5-plus
bash launch_sft_jobs.sh check-api gpt-6-astra

# 发起一条短请求，检查具体解码参数是否受模型支持，但不启动正式任务
bash launch_sft_jobs.sh preflight kimi-k3 greedy
bash launch_sft_jobs.sh preflight kimi-k3 pass8
bash launch_sft_jobs.sh preflight qwen3.5-plus greedy
bash launch_sft_jobs.sh preflight qwen3.5-plus pass8
bash launch_sft_jobs.sh preflight gpt-6-astra greedy
bash launch_sft_jobs.sh preflight gpt-6-astra pass8

# 后台生成；不同教师可并行，同一教师的两种解码建议依次执行
bash launch_sft_jobs.sh start kimi-k3 pass8
bash launch_sft_jobs.sh start deepseek-v4-pro greedy
bash launch_sft_jobs.sh start deepseek-v4-pro pass8
bash launch_sft_jobs.sh start qwen3.5-plus greedy
bash launch_sft_jobs.sh start qwen3.5-plus pass8
bash launch_sft_jobs.sh start gpt-6-astra greedy
bash launch_sft_jobs.sh start gpt-6-astra pass8

bash launch_sft_jobs.sh status
bash launch_sft_jobs.sh tail kimi-k3 pass8
```

关闭 FinalShell 不影响 `nohup` 后台任务。

## 严格校验和合并

生成结束后先合并同一模型已有的解码结果，再合并所有模型。若某个模型不支持
temperature=0，可以只运行 pass@8；`merge-model` 会直接在该模型每题的 8 条
候选中严格复验并选择最高质量的一条，无需伪造 greedy 结果：

```bash
bash launch_sft_jobs.sh merge-model kimi-k3
bash launch_sft_jobs.sh merge-model deepseek-v4-pro
bash launch_sft_jobs.sh merge-all
bash launch_sft_jobs.sh compare
```

`merge-model` 会先严格重放每条候选的工具 observation 和最终 SQL，再从同一
ID 的合格轨迹中选择一条质量最高的。`merge-all` 会再次在不同教师间按同样规则
选择每个 ID 的唯一轨迹并重审。

严格保留条件包括：

- 独立重放后 EX=True；
- XML 协议完整、标签闭合且无解析修复告警；
- 2～6 次工具调用，assistant 总轮数不超过 7；
- 最终 SQL 已由 `sqlite_query` 原样执行；
- `Verified call: k` 指向该次最终 SQL 调用；
- 工具 observation 可从数据库重放；
- 不出现 `<think>`、特权 SQL 提及或不可见 gold 字符串字面量。

逻辑有效性通过字段/工具足迹、observation 重放、最终 SQL 原样执行和 EX 复验进行
确定性检查；相关启发式告警写入审计文件，不冒充语义正确性的形式化证明。

## 产物

输出根目录：

`/data/keenSQL/dataset/sft_data/privileged_traces_sft_privileged_traces_kimi_ds`

每个模型、每种解码分别位于：

`models/<模型安全名>/<greedy|pass8>/`

- `raw_attempts.jsonl`：所有生成尝试、EX 和生成时审计，支持断点续跑。
- `accepted_sft.jsonl`：通过第一版生成时筛选的轨迹。
- `rejected_attempts.jsonl`：未通过轨迹及原因。
- `summary.json`：尝试数、通过数、命中 ID 和失败原因。
- `config.json`：输入、模型、URL、提示词和资产哈希，不包含 API Key。
- `strict_audit.jsonl`：严格复验的逐条结果。
- `strict_validated_sft.jsonl`：严格复验后可参与合并的轨迹。

日志和 PID：

- `logs/<模型安全名>_<解码>.log`
- `logs/<模型安全名>_<解码>.pid`

单模型最终结果：

`merged/by_model/<模型安全名>/sft_privileged_<模型安全名>.jsonl`

多模型最终结果：

`merged/all/sft_privileged_kimi_ds_merged.jsonl`

最终文件每个原始样本 ID 最多保留一条最高质量、严格 EX 正确的教师轨迹，可用于
与既有 Qwen 教师轨迹继续按 ID 合并。
