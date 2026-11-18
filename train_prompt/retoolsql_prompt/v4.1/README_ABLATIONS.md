# v4 ablations

- `ablate_no_verified_call.txt` — v4 prompt with the `Verified call: k` line and
  its rules removed. Run with a parser whose `verify_final` is still recorded,
  so the only difference is the prompt instruction.
- `no_lenient_tag` — no prompt variant. Run v4 with
  `--no-lenient-tag`, which disables the single-`</function>` repair.
