#!/usr/bin/env python3
"""In-run guard for a generation started by launch_jobs_v2.sh (borrowed from GAIA run_guard.sh).

Every --interval seconds:
  1. tripwire: scan_traces.scan() with the `guard` profile over the run's raw attempts;
     a NEW blocking anomaly freezes the run and writes <root>/FROZEN_ANOMALY.json.
     Resume only by hand after investigating:  run_guard.py resume --root R --model M
  2. balance floor: the run freezes (<root>/FROZEN_BALANCE.json) when OpenRouter
     limit_remaining falls below --floor PLUS the spend expected before the next two checks
     (burn rate measured from successive balance readings). A fixed floor alone failed on
     2026-11-16: at ~$3.5/min (48 concurrent requests) the balance went from $6.99 to the key's
     hard limit within one 60 s interval. It auto-resumes above floor + --resume-margin, but
     never while an anomaly freeze is in place.
The guard exits when every generator process has exited.

Freezing (GAIA lesson E36): processes are found from the launcher's pid files, never by
`pgrep -f` (which also matches shells whose command line contains the pattern). Each pid
gets SIGSTOP and the kernel state is checked; if a process does not reach state T it is
killed instead of being left to spend unguarded. A freeze/thaw self-test runs at start.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import scan_traces as S  # noqa: E402


def pid_files(root: Path, model: str, decoding: str) -> list[Path]:
    return sorted((root / "logs" / C.model_slug(model) / decoding).glob("chunk_*.pid"))


def proc_state(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat[stat.rfind(")") + 2]


def alive(pids: list[int]) -> list[int]:
    return [p for p in pids if proc_state(p) not in (None, "Z", "X")]


def read_pids(root: Path, model: str, decoding: str) -> list[int]:
    pids = []
    for path in pid_files(root, model, decoding):
        try:
            pids.append(int(path.read_text().strip()))
        except ValueError:
            continue
    return pids


def log(root: Path, message: str) -> None:
    line = f"{C.utc_now()} [guard] {message}"
    print(line, flush=True)
    with (root / "guard.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def freeze(root: Path, pids: list[int]) -> list[int]:
    stopped = []
    for pid in alive(pids):
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            continue
        for _ in range(20):
            if proc_state(pid) == "T":
                break
            time.sleep(0.1)
        if proc_state(pid) == "T":
            stopped.append(pid)
        else:
            log(root, f"pid {pid} did not stop (state {proc_state(pid)}); killing it")
            os.kill(pid, signal.SIGKILL)
    return stopped


def thaw(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


def self_test(root: Path, pids: list[int]) -> bool:
    live = alive(pids)
    if not live:
        return True
    stopped = freeze(root, live[:1])
    thaw(stopped)
    ok = bool(stopped) and proc_state(live[0]) != "T"
    log(root, f"self-test freeze/thaw on pid {live[0]}: {'ok' if ok else 'FAILED'}")
    return ok


def balance_reserve(prev: tuple[float, float] | None, now: float, remaining: float, interval: float) -> tuple[float, float]:
    """(burn USD/s, reserve USD): spend expected over the next two check intervals."""
    if prev is None or now <= prev[0]:
        return 0.0, 0.0
    burn = max(0.0, (prev[1] - remaining) / (now - prev[0]))
    return burn, burn * interval * 2


def should_freeze(remaining: float | None, floor: float, reserve: float) -> bool:
    return remaining is not None and remaining < floor + reserve


def cmd_watch(args) -> int:
    root, key = args.root, C.openrouter_key()
    for _ in range(30):  # the launcher writes pid files right after starting
        pids = read_pids(root, args.model, args.decoding)
        if pids:
            break
        time.sleep(2)
    if not pids:
        log(root, "no pid files found; nothing to guard")
        return 2
    log(root, f"guarding pids {pids}; floor ${args.floor}; interval {args.interval}s")
    if not self_test(root, pids):
        log(root, "self-test failed: killing the run rather than running unguarded")
        for pid in alive(pids):
            os.kill(pid, signal.SIGKILL)
        return 3
    limits = dict(S.PROFILES["guard"])
    seen: set[str] = set()
    prev_balance: tuple[float, float] | None = None
    anomaly_marker, balance_marker = root / "FROZEN_ANOMALY.json", root / "FROZEN_BALANCE.json"
    while alive(pids):
        time.sleep(args.interval)
        metrics = S.scan(root, expect_model=args.model)
        blocking = [b for b in S.judge(metrics, limits, "guard") if b.split(":")[0] not in seen]
        if blocking and not anomaly_marker.exists():
            stopped = freeze(root, pids)
            C.write_json(anomaly_marker, {"at": C.utc_now(), "blocking": blocking, "stopped": stopped,
                                          "metrics": {k: v for k, v in metrics.items() if k != "samples"},
                                          "samples": metrics.get("samples")})
            seen.update(b.split(":")[0] for b in blocking)
            log(root, f"FROZEN (anomaly): {blocking}; resume by hand after investigating")
            continue
        try:
            remaining = C.openrouter_balance(key)["limit_remaining"]
        except Exception as exc:  # noqa: BLE001
            log(root, f"balance check failed: {exc}")
            continue
        now = time.time()
        burn, reserve = balance_reserve(prev_balance, now, remaining, args.interval) if remaining is not None else (0.0, 0.0)
        if remaining is not None:
            prev_balance = (now, remaining)
        if should_freeze(remaining, args.floor, reserve) and not balance_marker.exists():
            stopped = freeze(root, pids)
            C.write_json(balance_marker, {"at": C.utc_now(), "limit_remaining": remaining, "burn_usd_per_min": round(burn * 60, 3),
                                          "reserve": round(reserve, 2), "stopped": stopped})
            log(root, f"FROZEN (balance ${remaining:.2f} < floor ${args.floor} + reserve ${reserve:.2f} at ${burn * 60:.2f}/min)")
        elif (balance_marker.exists() and not anomaly_marker.exists() and remaining is not None
              and remaining >= args.floor + args.resume_margin):
            thaw(pids)
            balance_marker.unlink()
            log(root, f"resumed (balance ${remaining:.2f})")
        if metrics.get("attempts"):
            log(root, f"attempts={metrics['attempts']} accepted={metrics['accepted']} "
                      f"cost=${metrics['cost_usd']} balance=${remaining} burn=${burn * 60:.2f}/min")
    log(root, "all generator processes exited; guard done")
    return 0


def cmd_resume(args) -> int:
    pids = read_pids(args.root, args.model, args.decoding)
    thaw(pids)
    for name in ("FROZEN_ANOMALY.json", "FROZEN_BALANCE.json"):
        marker = args.root / name
        if marker.exists():
            marker.rename(marker.with_name(f"{marker.stem}.resolved_{int(time.time())}.json"))
    log(args.root, f"manual resume of {pids}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("watch", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True, help="generation OUTPUT_ROOT")
        p.add_argument("--model", required=True)
        p.add_argument("--decoding", default="greedy")
        if name == "watch":
            p.add_argument("--interval", type=int, default=30)
            p.add_argument("--floor", type=float, default=15.0)
            p.add_argument("--resume-margin", type=float, default=20.0)
    args = parser.parse_args()
    return {"watch": cmd_watch, "resume": cmd_resume}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
