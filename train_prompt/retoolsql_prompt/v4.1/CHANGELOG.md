# v4.1 — changes from v4

v4.1 is a copy of `v4/`. Only the files listed here differ. `v4/` is untouched
and remains the prompt that existing datasets and checkpoints were built with.

## fewshot.json — example 0 rewritten

v4 example 0 answered "Which city has the most charter schools?" with
`WHERE charter = 1 AND city = 'ALAMEDA' ... GROUP BY city`, which can only
return ALAMEDA: the filter assumes the answer. v4.1 keeps what the episode
demonstrates (which-form single column, `ORDER BY ... LIMIT 1`, an empty
result repaired with `bm25_search_sqlite`) on a question that needs the filter:

- question: "Which district in St. Helena has the most charter schools?"
- schema: adds `district:TEXT`
- draft filters `city = 'ST. HELENA'` (uppercase, as Examples suggest) and
  returns 0 rows; bm25 on `city` finds the stored `'SAINT HELENA'`; the
  re-run returns one district and is cited as `Verified call: 3`.

Example 1 is unchanged. Changing the few-shot changes `prompt_hash`, so rows
generated with v4.1 must be exported with the v4.1 prefix.

## no_teacher_appendix.txt — new

Placeholder passed as `--teacher-prompt` by the no-gold easy-band run
(`train_prompt/sft_easy_nogold_v41/`). That run appends nothing to the user
message; the file exists so run configs record which teacher text was used.
