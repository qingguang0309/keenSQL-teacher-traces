# 工具默认参数校准（2026-11-04）

三个工具（`sqlite_query` / `sqlite_peek` / `bm25_search_sqlite`）的默认值，
依据 395 个训练数据库、2,928 张表的真实行数，以及 1,080 + 3,869 条已接受
SFT 轨迹和 5,821 条历史 SQL 的重放实验确定。

## 结论

| 参数 | 旧值 | 新值 | 依据 |
| --- | ---: | ---: | --- |
| `sqlite_query` `max_return_rows` | 100 | **200** | 44.0% 的表 ≤200 行；90.5% 的 gold 答案 ≤200 行；重放中 101–200 是唯一的中间档，200 以上直接跳到 >500 |
| `sqlite_peek` 扫描行数 `PEEK_SCAN_LIMIT` | 5,000 | **20,000** | 20,000 是表行数的 p95，94.9% 的表可全量扫描；截断的 peek 调用从 18.7% 降到 9.0% |
| `sqlite_peek` `limit` | 10 | **5** | 96% 的历史调用显式传 5（few-shot 也是 5）；4,000 字符预算本来就会把列表裁到 5/3/1 |
| `bm25_search_sqlite` `top_k` | 10 | **10（不变）** | 84% 的调用只返回 1 条；模型最终采用的字面量 98.4% 排第 1；降到 5 会丢掉一个 EX 成功的 rank-17 案例 |

未改动：`DEFAULT_MAX_OUTPUT_CHARS=4000`、`MAX_RETURN_ROWS_CAP=10000`、
`MAX_PEEK_LIMIT=500`、`MAX_TOP_K=100`、`VM_STEP_LIMIT=15,000,000`、
`BUSY_TIMEOUT_SECONDS=5`。

## 表行数统计

`tools/scan_table_rows.py` 只读扫描四个语料库（`bird_train` 69、`bird_dev` 11、
`omni` 150、`spider` 166），`all_dataset` 是上述数据库的副本，按 `db_id` 去重后跳过。
完整结果在 `database_tools/table_row_counts.json`。

| 分位 | p50 | p75 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 单表行数 | 299 | 1,000 | 5,268 | 20,662 | 1,056,320 | 129,131,916 |

| 行数上限 | 可全量覆盖的表 | 占 2,928 表 |
| ---: | ---: | ---: |
| 10 | 536 | 18.3% |
| 100 | 1,140 | 38.9% |
| **200** | **1,288** | **44.0%** |
| 500 | 1,795 | 61.3% |
| 1,000 | 2,231 | 76.2% |
| 5,000 | 2,627 | 89.7% |
| **20,000** | **2,778** | **94.9%** |
| 50,000 | 2,813 | 96.1% |

最大的表：`language_corpus` 129,131,916 行，`bike_share_1` 71,984,434 行，
`talkingdata` 32,473,067 行。对这些表任何返回上限都只是抽样，真正约束是
4,000 字符的 observation 预算。

## 训练经验

### 查询返回上限

已接受的 SFT 轨迹（`sft_privileged_merged.jsonl` 1,080 条、
`qwen36plus-pass8/accepted_sft.jsonl` 3,869 条）中：

| 指标 | merged | qwen36plus-pass8 |
| --- | ---: | ---: |
| `sqlite_query` 调用 | 6,377 | 23,640 |
| `row_count` p50 / p90 / p95 / p99 | 1 / 5 / 22 / 100 | 1 / 5 / 20 / 100 |
| `truncated=true` | 3.53% | 3.29% |
| `output_truncated=true` | 1.65% | 1.79% |
| 显式传 `max_return_rows` | 0.7% | 0.2% |

在 6,291 条查询观察中，99% 分位正好等于旧的 100，说明上限确实在约束一部分查询；但
`output_truncated` 只有约 1.7%，说明 4,000 字符预算在多数情况下更早生效。

另外，截断与失败**不相关**：EX 成功的 episode 有 15.4% 至少出现一次行截断，
失败 episode 只有 7.8%（成功轨迹工具调用更多）。所以调高上限是为了信息完整性，
不是因为它导致 EX 下降。

### 重放实验

把 6,859 条去重后的历史 `sqlite_query` 语句（来自 209 个库）用当前工具重放，
5,821 条成功执行；同一语句分别用 100/150/200/300/500 行上限跑一遍：

| 上限 | 行截断率 | 字符预算截断率 | 平均 payload |
| ---: | ---: | ---: | ---: |
| 100 | 8.45% | 4.29% | 518 字符 |
| 150 | 7.82% | 5.57% | 546 字符 |
| **200** | **7.44%** | **6.18%** | **559 字符** |
| 300 | 7.35% | 6.82% | 569 字符 |
| 500 | 7.35% | 7.32% | 573 字符 |

上限从 100 提到 200 时：

- 240 条（4.1%）查询返回更多行；
- 59 条不再被行上限截断；
- 181 条仍在行上限内截断，但多看到了一些行；
- 110 条新触发 4,000 字符预算截断；
- 平均只多 41 字符。

关键结构：101–200 行有 59 条，201–300 行只有 5 条，301–500 行为 0，
其余 181 条都超过 500 行。结果是**双峰的**——验证型查询返回 1–100 行，
列举型查询直接超过 500 行。200 是唯一的自然拐点；再往上只会增加预算截断，
不会多覆盖中间规模的查询。

### `sqlite_peek`

- 按调用次数统计，91.5% 的 peek 命中 5,000 行扫描上限（同一个大表被反复
  peek，如 `flights`）。
- 按去重后的 (db, 参数) 调用重放 444 条：5,000 行时 18.7% 被截断，20,000 行
  降到 9.0%，50,000 行降到 6.5%。
- Distinct 估计显著改善：`bike_share_1.trip` 第 3 列 5,000 行时
  `distinct=3,191`，20,000 行时 12,320，50,000 行时 29,539；
  `talkingdata.app_all` 从 4,803 → 47,836。5,000 行时 `distinct` 常常等于
  扫描行数，模型会把"抽样上限"误读成真实基数。
- 成本：三列 peek 在最大表上 5,000→20,000 行约 0.01s→0.1s；全列（35–55 列）
  peek 从 0.2–0.46s 增到 0.8–1.8s，仍在 30s 墙钟预算内。
- `limit`：96% 的调用显式传 5，few-shot 示例也是 5；`_fit_peek_budget` 在
  4,000 字符预算下按 5/3/1 逐级裁剪。默认 10 实际从未成为有效上界。

### `bm25_search_sqlite`

7,534 条 BM25 观察（5.2% 的调用返回 >10 条）：

| 返回条数 | 占比 |
| --- | ---: |
| 1 | 84.1% |
| 2 | 3.8% |
| 3 | 1.8% |
| 10（BM25 列表满） | 4.8% |
| 20（BM25 + regex 两个列表都满） | 0.3% |

把 BM25 返回值和最终 SQL 对齐后，模型实际采用的 7,820 个字面量里：

- 98.4% 排在第 1 位；
- 1.6% 排第 2 位及以后；
- 只有 20 条排第 4 位及以后，7 条排第 6 位及以后；
- rank ≥5 的 7 条里，去重后只有 1 个不同案例（`olympics`，rank 17，
  EX 成功）。它是 regex-fallback 列表的第 8 条，`top_k=5` 会把它裁掉。

因此 `top_k=10` 保留全部有用召回，且 84% 的调用根本用不到 10 条；维持 10。

## 改动

代码与文档：

- `database_tools/sqlite_tools.py`：四个常量与工具 schema 的 `default`/描述。
- `train_prompt/retoolsql_prompt/v4/system_prompt.txt`：
  `max_return_rows=200`、`limit=5`、`first 20,000 rows`。
- `train_prompt/retoolsql_prompt/v4/ablate_no_verified_call.txt`：同上。
- `train_prompt/retoolsql_prompt/retoolsql_prompt.txt`：同步 v4 的副本。
- `database_tools/prompts.py`：内置 fallback 的系统提示。
- `database_tools/README.md`、`tools/scan_table_rows.py`、
  `database_tools/table_row_counts.json`。
- `train_prompt/retoolsql_prompt/validate_prompt.py` 与
  `train_prompt/test_retoolsql_prompt.py` 的文档值检查。

测试：

- `database_tools/tests/test_sqlite_tools.py` 的 `big` fixture 改为
  `PEEK_SCAN_LIMIT + 1,000` 行，扫描上限测试改名并跟随常量。
- `database_tools/tests/test_prompts.py` 的签名与扫描行数断言。

## 影响

- `PROMPT_HASH` 从 `aa5dd816…` 变为
  `0fe1db71119901cc17be8fbe227953f69a718d53ada73c3ce37d877bafb81434`。
  这个最终值同时包含两处修正：4.1 节工具调用骨架的显式
  `max_return_rows=200`，以及 few-shot `stored_format_peek` 在 20,000 行扫描上限下
  `truncated=false`。现有 SFT/RFT parquet 的 `prompt_hash` 与 trace 的 system prompt
  都对应旧值，`sft_train/preflight.py` 的 `train_prompt_hash` 检查会失败，
  `detect_prompt_messages` 也会拒绝旧 trace。
- 工作区里的 `data/sft.parquet`、`data/sft_train.parquet`、
  `data/sft_val.parquet` 是 2026-11-04 16:41 由
  `sft_easy_teacher_recovery_2033/build_final_sft_dataset.py` 用旧提示词生成的；
  这些轨迹需要按 `sft_train/repair_sft_v3.py` 重放工具调用并定向重生成不一致样本，
  才能与新提示词/新工具默认参数一致。
- `output/sft/audit/probe_report.json`（10-27）相对新数据集已经过期，
  `test/tests/test_probe_fixed_set.py::test_reported_probe_matches_the_selection`
  会失败；这与本次工具参数改动无关，是数据集重建导致的选择集漂移。

## 复现

```bash
# 表行数
python tools/scan_table_rows.py --json database_tools/table_row_counts.json

# 历史 SQL 重放（上限对比）：对 sft_privileged_merged.jsonl 与
# qwen36plus-pass8/accepted_sft.jsonl 提取 <tool_call> 里的 sql，
# 按 (db_id, sha256(sql)) 去重后用 sqlite_query(..., max_return_rows=cap,
# max_output_chars=DEFAULT_MAX_OUTPUT_CHARS) 重跑。

# prompt 与默认值一致性
python train_prompt/retoolsql_prompt/validate_prompt.py
python -m pytest database_tools/tests -q
```
