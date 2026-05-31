"""Resume state for Sayobot batch downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audio2map.utils.paths import collector_dir, raw_dir

STATE_PATH = collector_dir() / "state.json"


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8-sig"))
    return {"offset": 0, "completed": [], "failed": {}, "skipped_no_4k": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def print_state_stats(state: dict[str, Any]) -> None:
    n_ok = len(state.get("completed", []))
    n_skip = len(state.get("skipped_no_4k", []))
    n_fail = len(state.get("failed", {}))
    total = n_ok + n_skip + n_fail
    dataset = raw_dir()
    dataset_n = (
        sum(1 for d in dataset.iterdir() if d.is_dir() and d.name.isdigit())
        if dataset.is_dir()
        else 0
    )

    def pct(n: int) -> str:
        return "  0.0%" if total == 0 else f"{100.0 * n / total:5.1f}%"

    w = 40
    print(f"\n{'state.json stats':^{w}}\n{'-' * w}")
    print(f"  completed      {n_ok:>5}    {pct(n_ok)}")
    print(f"  skipped_no_4k  {n_skip:>5}    {pct(n_skip)}")
    print(f"  failed         {n_fail:>5}    {pct(n_fail)}")
    print(f"{'-' * w}\n  total_handled  {total:>5}    100.0%")
    print(f"  offset         {int(state.get('offset', 0)):>5}")
    print(f"  raw dirs       {dataset_n:>5}\n")


def should_skip(sid: int, state: dict[str, Any]) -> bool:
    return sid in state.get("completed", []) or sid in state.get("skipped_no_4k", [])


def apply_result(sid: int, status: str, state: dict[str, Any], err: str | None = None) -> None:
    key = str(sid)
    if status in ("completed", "already_complete"):
        state["completed"] = sorted(set(state["completed"]) | {sid})
        state["failed"].pop(key, None)
        if sid in state["skipped_no_4k"]:
            state["skipped_no_4k"].remove(sid)
    elif status == "skipped_no_4k":
        if sid not in state["skipped_no_4k"]:
            state["skipped_no_4k"].append(sid)
        state["failed"].pop(key, None)
        if sid in state["completed"]:
            state["completed"].remove(sid)
    elif status == "failed" and err:
        state["failed"][key] = err
        if sid in state["completed"]:
            state["completed"].remove(sid)


def persist(state: dict[str, Any], offset: int) -> None:
    state["offset"] = offset
    state["completed"] = sorted(set(state["completed"]))
    state["skipped_no_4k"] = sorted(set(state["skipped_no_4k"]))
    save_state(state)
