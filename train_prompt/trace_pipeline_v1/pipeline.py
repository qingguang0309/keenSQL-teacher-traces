#!/usr/bin/env python3
"""Teacher-trace pipeline: gate -> generate -> audit -> adjudicate -> resample -> release.

Standing rule (2026-11-12): no teacher trace is used for training before it passes the audit
and adjudication stages below. Each stage writes its result under the run root and the next
stage refuses to start when the previous gate did not pass.

  prepare   inputs for a batch: --parent-input + (--ids-file | --n) -> <root>/inputs
  gate      preflight.py (models, params, live probes, inputs, balance)  -> <root>/gate/preflight.json
  smoke     --n questions through the real generator, then scan_traces --profile smoke
                                                                         -> <root>/gate/smoke_scan.json
  generate  launch_jobs_v2.sh start-all + run_guard.py watch (needs gate and smoke passed)
  merge     strict audit + merge (v4.1 auditor) and a full-run scan       -> <root>/gate/run_scan.json
  audit     L3a build, planted errors, two judges, aggregate (calibration gate), L3b, final
                                                                         -> <audit_dir>/final_report.json
  resample  select failed questions and prepare K sample roots (each is run with
            gate/smoke/generate/merge/audit like a batch)
  release   best trace per question, dev decontamination, manifest       -> <out>/release_manifest.json

Keys come from the environment only (OPENROUTER_API_KEY); nothing here writes them anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

PY = os.environ.get("PY", "/data/keenSQL/.venv/bin/python")
JUDGE_MAX_TOKENS = {"kimi": 24000, "deepseek": 48000}  # deepseek hit length at 24k in the 395 batch


def sh(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> int:
    print(f"[pipeline] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env={**os.environ, **(env or {})}).returncode
    if check and rc != 0:
        raise SystemExit(f"[pipeline] step failed (exit {rc}): {' '.join(cmd[:3])}")
    return rc


def launcher(root: Path, *args: str, workers: int = 3, extra_env: dict[str, str] | None = None,
             check: bool = True) -> int:
    env = {"SFT_EASY_NOGOLD_OUTPUT_ROOT": str(root), "SFT_API_WORKERS": str(workers),
           "SFT_REQUIRE_BOTH_MODES": "0", **(extra_env or {})}
    return sh(["bash", str(C.LAUNCHER), *args], env=env, check=check)


def gate_ok(root: Path, name: str) -> bool:
    path = root / "gate" / name
    return path.is_file() and bool(json.loads(path.read_text()).get("passed"))


def wait_for_generators(root: Path, model: str, decoding: str, poll: int = 30) -> None:
    import run_guard as G
    pids = G.read_pids(root, model, decoding)
    print(f"[pipeline] waiting for generator pids {pids}", flush=True)
    while G.alive(pids):
        time.sleep(poll)
    print("[pipeline] generators exited", flush=True)


def temperature_env(args) -> dict[str, str]:
    return {"SFT_TEMPERATURE_OVERRIDE": str(args.temperature)} if getattr(args, "temperature", None) is not None else {}


# ------------------------------------------------------------------ stages

def cmd_prepare(args) -> int:
    parent = json.loads(Path(args.parent_input).read_text(encoding="utf-8"))
    if args.ids_file:
        wanted = [line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()]
        by_id = {str(r["question_id"]): r for r in parent}
        missing = [q for q in wanted if q not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} ids not in parent input, e.g. {missing[:5]}")
        records, selection = [by_id[q] for q in wanted], f"ids from {args.ids_file}"
    else:
        records = C.stratified_sample(parent, args.n, args.seed)
        selection = f"stratified sample n={args.n} seed={args.seed} over (source, difficulty)"
    manifest = C.prepare_inputs(records, args.root, selection=selection,
                                parent_input=Path(args.parent_input), chunk_size=args.chunk_size)
    print(f"[pipeline] prepared {manifest['selected_records']} records in {manifest['chunks']} chunks under {args.root}/inputs")
    return 0


def cmd_gate(args) -> int:
    cmd = [PY, str(HERE / "preflight.py"), "--teacher", args.teacher,
           "--input", str(args.root / "inputs/easy_current.json"),
           "--floor", str(args.floor), "--budget", str(args.budget),
           "--out", str(args.root / "gate/preflight.json")]
    for waiver in args.waive:
        cmd += ["--waive", waiver]
    return sh(cmd, check=False)


def cmd_smoke(args) -> int:
    if not gate_ok(args.root, "preflight.json"):
        raise SystemExit("[pipeline] preflight has not passed; run `gate` first")
    smoke_root = args.root / "smoke"
    records = json.loads((args.root / "inputs/easy_current.json").read_text(encoding="utf-8"))
    if not (smoke_root / "inputs/manifest.json").is_file():
        C.prepare_inputs(C.stratified_sample(records, args.n, args.seed), smoke_root,
                         selection=f"smoke sample n={args.n}", parent_input=args.root / "inputs/easy_current.json",
                         chunk_size=args.n)
    launcher(smoke_root, "start-all", args.teacher, args.decoding, workers=args.workers,
             extra_env=temperature_env(args))
    wait_for_generators(smoke_root, args.teacher, args.decoding, poll=10)
    return sh([PY, str(HERE / "scan_traces.py"), str(smoke_root), "--profile", "smoke",
               "--expect-model", args.teacher, "--out", str(args.root / "gate/smoke_scan.json")], check=False)


def cmd_generate(args) -> int:
    for name in ("preflight.json", "smoke_scan.json"):
        if not gate_ok(args.root, name) and not args.skip_gate_check:
            raise SystemExit(f"[pipeline] gate {name} has not passed; refusing to generate")
    launcher(args.root, "start-all", args.teacher, args.decoding, workers=args.workers,
             extra_env=temperature_env(args))
    guard_log = (args.root / "guard.out").open("a")
    guard = subprocess.Popen([PY, str(HERE / "run_guard.py"), "watch", "--root", str(args.root),
                              "--model", args.teacher, "--decoding", args.decoding,
                              "--floor", str(args.floor), "--interval", str(args.guard_interval)],
                             stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
    (args.root / "guard.pid").write_text(str(guard.pid))
    print(f"[pipeline] run guard pid {guard.pid}; log {args.root}/guard.log", flush=True)
    if args.wait:
        wait_for_generators(args.root, args.teacher, args.decoding)
        guard.wait(timeout=600)
    return 0


def merged_path(root: Path, teacher: str) -> Path:
    slug = C.model_slug(teacher)
    return root / "merged/by_model" / slug / f"sft_easy_teacher_{slug}.jsonl"


def cmd_merge(args) -> int:
    launcher(args.root, "merge-model", args.teacher)
    sh([PY, str(HERE / "scan_traces.py"), str(args.root), "--profile", "guard",
        "--expect-model", args.teacher, "--out", str(args.root / "gate/run_scan.json")], check=False)
    path = merged_path(args.root, args.teacher)
    print(f"[pipeline] merged strict-audited traces: {path} ({sum(1 for _ in open(path))} rows)")
    return 0


def cmd_audit(args) -> int:
    a = args.out
    a.mkdir(parents=True, exist_ok=True)
    sft = args.sft or merged_path(args.root, args.teacher)
    audit = [PY, str(HERE / "audit_l3.py")]
    clean, planted = a / "clean_items.jsonl", a / "corrupted.jsonl"
    if not clean.is_file():
        sh(audit + ["build", "--inputs", f"{args.run_name}={sft}", "--out", str(clean)])
    if not planted.is_file():
        sh(audit + ["corrupt", "--items", str(clean), "--out", str(planted), "--rate", str(args.rate)])
    items = a / "items.jsonl"
    if not items.is_file():
        rows = list(C.iter_jsonl(clean)) + list(C.iter_jsonl(planted))
        random.Random(args.seed).shuffle(rows)  # judges must not see planted items as a block
        C.write_jsonl(items, rows)
        (a / "items.jsonl.sources.json").write_text((Path(str(clean) + ".sources.json")).read_text())
    procs = []
    for judge in ("kimi", "deepseek"):
        cmd = audit + ["judge", "--judge", judge, "--items", str(items), "--out", str(a / f"{judge}.jsonl"),
                       "--workers", str(args.workers), "--max-tokens", str(JUDGE_MAX_TOKENS[judge])]
        print(f"[pipeline] $ {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, stdout=(a / f"{judge}.log").open("a"), stderr=subprocess.STDOUT))
    if any(p.wait() for p in procs):
        raise SystemExit("[pipeline] a judge run failed (circuit breaker?); see kimi.log / deepseek.log; rerun to resume")
    rc = sh(audit + ["aggregate", "--items", str(items), "--dir", str(a)], check=False)
    if rc == 3:
        raise SystemExit("[pipeline] calibration gate failed: judges missed planted errors; batch not trusted")
    if rc:
        raise SystemExit(f"[pipeline] aggregate failed ({rc})")
    l3b = [PY, str(HERE / "audit_l3b.py")]
    sh(l3b + ["run", "--items", str(items), "--l3a-dir", str(a), "--out", str(a / "l3b.jsonl"),
              "--workers", str(args.workers)])
    sh(l3b + ["final", "--items", str(items), "--l3a-dir", str(a), "--l3b", str(a / "l3b.jsonl"), "--out-dir", str(a)])
    return 0


def cmd_resample(args) -> int:
    plan = args.out / "select.json"
    cmd = [PY, str(HERE / "resample.py"), "select", "--parent-input", str(args.root / "inputs/easy_current.json"),
           "--out", str(plan), "--gen-root", str(args.root)]
    for audit_dir in args.audit:
        cmd += ["--audit-dir", str(audit_dir)]
    sh(cmd)
    return sh([PY, str(HERE / "resample.py"), "prepare", "--plan", str(plan), "--out", str(args.out),
               "--samples", str(args.samples)])


def cmd_release(args) -> int:
    cmd = [PY, str(HERE / "release.py"), "--out-dir", str(args.out)]
    for audit_dir in args.audit:
        cmd += ["--audit-dir", str(audit_dir)]
    for root in args.gen_root:
        cmd += ["--gen-root", str(root)]
    if args.consistency:
        cmd += ["--consistency", str(args.consistency)]
    return sh(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def run_args(p, teacher=True):
        p.add_argument("--root", type=Path, required=True, help="batch OUTPUT_ROOT")
        if teacher:
            p.add_argument("--teacher", required=True)

    p = sub.add_parser("prepare"); run_args(p, teacher=False)
    p.add_argument("--parent-input", type=Path, required=True)
    p.add_argument("--ids-file", type=Path, default=None)
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--chunk-size", type=int, default=34)
    p = sub.add_parser("gate"); run_args(p)
    p.add_argument("--budget", type=float, required=True, help="expected USD spend of the batch (gen + audit)")
    p.add_argument("--floor", type=float, default=15.0)
    p.add_argument("--waive", action="append", default=[])
    p = sub.add_parser("smoke"); run_args(p)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--decoding", default="greedy")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--temperature", type=float, default=None)
    p = sub.add_parser("generate"); run_args(p)
    p.add_argument("--decoding", default="greedy")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--floor", type=float, default=15.0)
    p.add_argument("--guard-interval", type=int, default=30)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--skip-gate-check", default=None, metavar="REASON")
    p = sub.add_parser("merge"); run_args(p)
    p = sub.add_parser("audit"); run_args(p)
    p.add_argument("--run-name", required=True, help="prefix of audit ids, e.g. batch2")
    p.add_argument("--out", type=Path, required=True, help="audit dir")
    p.add_argument("--sft", type=Path, default=None, help="default: the merged strict-audited file")
    p.add_argument("--rate", type=float, default=0.15, help="planted-error share")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=20261112)
    p = sub.add_parser("resample"); run_args(p, teacher=False)
    p.add_argument("--audit", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=3)
    p = sub.add_parser("release")
    p.add_argument("--audit", type=Path, action="append", required=True)
    p.add_argument("--gen-root", type=Path, action="append", default=[])
    p.add_argument("--consistency", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return {"prepare": cmd_prepare, "gate": cmd_gate, "smoke": cmd_smoke, "generate": cmd_generate,
            "merge": cmd_merge, "audit": cmd_audit, "resample": cmd_resample,
            "release": cmd_release}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
