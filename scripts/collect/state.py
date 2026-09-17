"""Resume state for Sayobot batch downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audio2map.utils.paths import collector_dir, raw_dir


def state_path() -> Path:
    return collector_dir() / "state.json"


_SID_LISTS = ("completed", "skipped_no_4k", "skipped_no_mania", "skipped_class")


def _sid_set(value: Any) -> set[int]:
    if not value:
        return set()
    return {int(x) for x in value}


def _empty_state() -> dict[str, Any]:
    return {
        "offset": 0,
        "completed": set(),
        "failed": {},
        "skipped_no_4k": set(),
        "skipped_no_mania": set(),
        "skipped_class": set(),
    }


def _disk_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "offset": int(state.get("offset", 0) or 0),
        "completed": sorted(_sid_set(state.get("completed"))),
        "skipped_no_4k": sorted(_sid_set(state.get("skipped_no_4k"))),
        "skipped_no_mania": sorted(_sid_set(state.get("skipped_no_mania"))),
        "skipped_class": sorted(_sid_set(state.get("skipped_class"))),
        "failed": {str(k): v for k, v in (state.get("failed") or {}).items()},
    }


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.is_file():
        return _empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"corrupt collector state: {path}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"corrupt collector state: {path}")
    out = _empty_state()
    out["offset"] = int(data.get("offset", 0) or 0)
    for name in _SID_LISTS:
        raw = data.get(name, [])
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise SystemExit(f"corrupt collector state: {path}")
        out[name] = _sid_set(raw)
    failed = data.get("failed") or {}
    if not isinstance(failed, dict):
        raise SystemExit(f"corrupt collector state: {path}")
    out["failed"] = {str(k): v for k, v in failed.items()}
    return out


def save_state(state: dict[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_disk_payload(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def print_state_stats(state: dict[str, Any]) -> None:
    n_ok = len(state.get("completed") or [])
    n_no_4k = len(state.get("skipped_no_4k") or [])
    n_no_mania = len(state.get("skipped_no_mania") or [])
    n_class = len(state.get("skipped_class") or [])
    n_fail = len(state.get("failed") or {})
    total = n_ok + n_no_4k + n_no_mania + n_class + n_fail
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
    print(f"  completed         {n_ok:>5}    {pct(n_ok)}")
    print(f"  skipped_no_4k     {n_no_4k:>5}    {pct(n_no_4k)}")
    print(f"  skipped_no_mania  {n_no_mania:>5}    {pct(n_no_mania)}")
    print(f"  skipped_class     {n_class:>5}    {pct(n_class)}")
    print(f"  failed            {n_fail:>5}    {pct(n_fail)}")
    print(f"{'-' * w}\n  total_handled     {total:>5}    100.0%")
    print(f"  offset            {int(state.get('offset', 0)):>5}")
    print(f"  raw dirs          {dataset_n:>5}\n")


def should_skip(sid: int, state: dict[str, Any]) -> bool:
    return any(sid in state.get(name, ()) for name in _SID_LISTS)


def _clear_sid(state: dict[str, Any], sid: int) -> None:
    for name in _SID_LISTS:
        state[name].discard(sid)
    state["failed"].pop(str(sid), None)


def apply_result(sid: int, status: str, state: dict[str, Any], err: str | None = None) -> None:
    if status in ("completed", "already_complete"):
        _clear_sid(state, sid)
        state["completed"].add(sid)
    elif status in ("skipped_no_4k", "skipped_no_mania", "skipped_class"):
        _clear_sid(state, sid)
        state[status].add(sid)
    elif status == "failed" and err:
        _clear_sid(state, sid)
        state["failed"][str(sid)] = err


def persist(state: dict[str, Any], offset: int) -> None:
    state["offset"] = int(offset)
    save_state(state)
