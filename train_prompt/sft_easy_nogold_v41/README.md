# 无特权（no-gold）easy 档轨迹生成，prompt v4.1

为 easy 档 9,119 题重新生成 SFT 示范轨迹。生成时**教师看不到 Gold SQL**，Gold 只用于事后的 EX 判分。背景和审核流程见本地文档 `轨迹审核Agent执行方案.md`。

## 与特权 easy-teacher 流程的差异

| 项 | 特权流程（`../sft_easy_teacher_traces`） | 本流程 |
|---|---|---|
| 教师 prompt | 学生 prompt + `sft_teacher_prompt.txt`（含 Gold SQL） | **只有学生 prompt**，不附加任何内容 |
| prompt 版本 | v4 | **v4.1**（修正了 few-shot 例 0，见 `retoolsql_prompt/v4.1/CHANGELOG.md`） |
| 最少工具调用次数 | 2 | **1**（首稿执行正确即可交付，见 system prompt §7） |
| 输出的 `data_variant` | `easy_teacher` | **`easy`**（`sft_train/import_traces.py` 对这一类免做 gold 泄漏检查） |
| 其余 | 生成器、API 适配层、EX 判分、协议检查、断点续跑 | 完全相同，复用原代码，没有改动 |

实现方式与项目里其他适配器一致：`build_nogold_traces.py` 用 importlib 加载 `../sft_easy_teacher_traces/build_easy_teacher_traces.py`，然后替换三处：

- `build_prompts`：不附加 Gold；
- `make_sft_record`：标记 `data_variant=easy`、`teacher_mode=nogold`；
- `MIN_TEACHER_TOOL_CALLS`：改为 1。

同时把 `prompt_builder` 的路径固定到 v4.1。**核心生成器和 v4 目录都没有被修改。**

## 目录

| 路径 | 内容 |
|---|---|
| `build_nogold_traces.py` | 适配器 |
| `launch_nogold_jobs.sh` | 原 launcher 的副本：只改了 SCRIPT、PROMPT、OUTPUT_ROOT 和 verify 的参数；禁用了 prepare / extract-existing / combine-current |
| 数据 `/data/keenSQL/dataset/m_schema/sft_data/easy_nogold_v41/` | `inputs/` 是 `easy_teacher_traces/inputs` 的副本（9,119 题，8 块，manifest 路径已改写，哈希一致）；`models/`、`logs/`、`merged/` 由运行时生成 |

## 运行

```bash
cd /data/keenSQL/train_prompt/sft_easy_nogold_v41
export QWEN_API_KEY='...'

bash launch_nogold_jobs.sh verify
bash launch_nogold_jobs.sh dry-run qwen3.6-plus greedy 0   # 不调用模型
bash launch_nogold_jobs.sh preflight qwen3.6-plus greedy

export SFT_API_WORKERS=2
bash launch_nogold_jobs.sh start-all qwen3.6-plus greedy   # 第一轮：全部 9,119 题
bash launch_nogold_jobs.sh status
bash launch_nogold_jobs.sh merge-model qwen3.6-plus        # greedy 跑完后合并
```

greedy 跑完后，对未通过的题再跑 pass8。需要先用 greedy 的拒绝结果生成一份新的输入块，这一步尚未实现。

## 2026-11-10 已完成的离线验证（未调用模型）

- `launch_nogold_jobs.sh verify` 通过：输入哈希、分块覆盖、数据库与 schema 齐全。
- `retoolsql_prompt/validate_prompt.py` 在 v4.1 上全部通过。few-shot 从 1,440 增加到 1,552 token（上限 1,600）。
- dry-run：`prompt_version=v4.1`、`teacher_mode=nogold`、`teacher_appendix_tokens=0`；sample_id 与之前的运行一致。
- 对 9,119 题全部离线构建 prompt：
  - 9,119 条全部构建成功，0 条超出 schema 预算；
  - 教师 prompt 与学生 prompt 完全相同；
  - 不含教师标记；
  - 任何一条 prompt 里都没有出现 Gold SQL。
- `make_sft_record` 输出：`data_variant=easy`、`prompt_version=v4.1`、`teacher_mode=nogold`。

## 2026-11-10 试点：GPT-5.6 Sol 生成 500 题（在 aws-asia 上运行）

**为什么在 aws-asia 上运行**：从集群通过 OpenRouter 调用 OpenAI、Anthropic、Google 的模型时，会返回 403 "not available in your region"。所以生成在公司服务器 aws-asia 上进行。

**aws-asia 的环境**：

- 目录布局与集群相同（`/data/keenSQL`）。代码、M-Schema cache、输入，以及试点涉及的 128 个库（32.5 GB，1,169 个文件，大小已逐一核对）都复制了过去。
- `.venv` 用 uv 安装：Python 3.11.17、SQLite 3.53.1，与集群相同；openai 3.35.1、sqlglot 30.23.0、tiktoken 0.14.0。
- 500 题的 prompt 在两边**逐字节一致**。

**本次新增**：

- `launch_nogold_jobs.sh`：模型名中带 `/` 的，自动走 OpenRouter，key 从 `OPENROUTER_API_KEY` 读取。
- `audit_nogold_traces.py`：严格审核的工具调用次数放宽到 1–6 次；只调用 1 次的样本保留 `single_tool_call_allowed_nogold` 警告。
- GPT-5.6 Sol 不支持 temperature 参数，OpenRouter 会忽略请求中的 `temperature=0`。所以这里的 "greedy" 实际上用的是模型默认的采样设置。

**运行方式**：

```bash
# 在 aws-asia 上
export OPENROUTER_API_KEY='...'
export SFT_EASY_NOGOLD_OUTPUT_ROOT=/data/keenSQL/dataset/m_schema/sft_data/easy_nogold_v41_pilot500_gpt56sol
SFT_API_WORKERS=3 bash launch_nogold_jobs.sh start-all openai/gpt-5.6-sol greedy
SFT_REQUIRE_BOTH_MODES=0 bash launch_nogold_jobs.sh merge-model openai/gpt-5.6-sol
```

**结果**：

| 项 | 数值 |
|---|---|
| 生成阶段通过 | 355 / 500（71.0%） |
| 严格审核后（重放工具 + 重算 EX） | **350 条**，结果在 `merged/by_model/openai-gpt-5.6-sol/`，已同步回集群 |
| 主要拒绝原因 | EX 错误 144；格式错误 68（模型在一轮输出里自己编出后续的工具结果和最终答案） |
| 花费 | $26.86，约 $0.054/题 |

**质量指标**（与 v3 特权轨迹对比）：

| 指标 | 本次 350 条 | v3 特权 easy |
|---|---:|---:|
| 最终 SQL 与 Gold 相同 | 4.3% | 47.6% |
| 别名映射与 Gold 相同 | 0.0% | 85.5% |
| 观察驱动的修复 | 10.3% | 3.8% |
| 只调用 1 次工具 | 86.3% | 0%（规则要求至少 2 次） |
| 用过 peek / bm25 | 4.3% | 69.3% |

## 2026-11-10 修复：每轮只保留第一个工具调用

**问题**：试点中有 67 题（13.4%）因格式错误被拒：

- 42 题：同一轮里既有工具调用又有最终答案。
- 25 题：同一轮里有 2 个工具调用。

原因是模型写完 `</tool_call>` 后没有停下，接着自己编了 observation，又继续往下写。

**修复**（只改 `build_nogold_traces.py`，上游共享脚本不动）：

1. 每轮输出在第一个 `</tool_call>` 处截断，后面的内容丢弃，然后执行真实的工具调用，把真实结果交给模型。被丢弃的部分只是模型对数据库返回结果的猜测，不包含真实数据，所以截断不会损失有效信息。每次截断都会在 stderr 打一行 `[nogold-stop]`。
2. 请求里同时带上 `stop=["</tool_call>"]`：
   - 支持 `stop` 的模型会直接停下，省掉这部分 token；这时代码会把被停止序列吃掉的 `</tool_call>` 补回来。
   - GPT-5.6 Sol 在 OpenRouter 的参数列表里没有 `stop`，会被静默忽略，所以对它起作用的是第 1 步的截断。
   - 如果某个 API 直接拒绝 `stop` 字段，设置 `SFT_NOGOLD_SEND_STOP=0` 即可。
3. 因 max_tokens 截断而没有闭合的工具调用保持原样，仍会按原规则判为失败。

**验证（离线，未调用 API）**：

- 把 67 条格式错误的回复在第一个 `</tool_call>` 处截断后重新解析，67 条全部变成合法的工具调用。
- 用真实数据库把其中一题（bird_00608）完整重放了一遍：第一轮在 `</tool_call>` 处结束，编造的 observation 没有保留，真实查询结果正常返回，episode 没有错误。

**注意**：上面那次 500 题试点用的是修复前的代码。修复后这 67 题需要重新生成，才能知道其中有多少能真正通过 EX 检查。

## 2026-11-10 重跑：试点中 67 题格式错误（修复后）

修复后，把试点中因格式错误被拒的 67 题，在 aws-asia 上用 GPT-5.6 Sol 重新生成了一遍。

- 结果目录：`dataset/m_schema/sft_data/easy_nogold_v41_stopfix67_gpt56sol/`，两台服务器上都有，内容一致。
- 运行时只换了输出目录（`SFT_EASY_NOGOLD_OUTPUT_ROOT=.../easy_nogold_v41_stopfix67_gpt56sol`），命令与试点相同。

| 项 | 数值 |
|---|---|
| 生成阶段通过 | 47 / 67（70.1%） |
| **严格审核后** | **45 条** |
| 截断次数 | 28 个回合（日志里的 `[nogold-stop]` 行） |
| 仍有的格式错误 | 0（编造后续内容的问题已消除） |
| 其他失败 | EX 错误 16；工具调用超过 6 次的上限 4（都是真实的反复查询）；严格审核剔除 2（`temporally_unseen_gold_string_literal`） |
| 花费 | 约 $4.4（OpenRouter 用量计数有延迟） |

**质量指标**：

- 最终 SQL 与 Gold 相同的比例为 8.9%（共 45 条，其中 4 条）。
- 只调用 1 次工具的占 71.1%。
- 用过 peek / bm25 的占 11.1%。
- 修复占 17.8%，指查询出错或返回空结果之后，模型又发起了新的查询。

这些题本来就是模型"想多查几步"的题，所以和试点整体相比，单次调用的比例更低，修复更多。

**与试点合计**：500 题中，**395 条**通过了严格审核（79.0%）。

## 2026-11-10 更正：修复率里混入了 HTML 转义的自我修正

把 395 条轨迹拉回本地逐条查看时发现：有 **29 条（7.3%）**，GPT-5.6 Sol 把 SQL 中的 `>=`、`<` 写成了 `&gt;=`、`&lt;`，执行时触发了 `validation_error`，下一轮才改回正常写法。最终答案里都没有转义，所以这些样本通过了审核。

之前报告的修复率（试点 10.6%、重跑 17.8%）把这些样本也算了进去。剔除后，**真正由数据驱动的修复有 16 / 395 条（4.1%）**，与 v3 的 3.8% 持平。

**处理建议**：

1. 训练前先筛掉这 29 条，否则学生会学到"先写错格式再改"的习惯。
2. 以后生成时，在截断那一步把转义还原后再解析。
