"""Batch download osu! mania 4K beatmaps from Sayobot."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from audio2map.osu.parser import audio_filename, is_mania_4k
from audio2map.utils.paths import collector_dir, raw_dir
from scripts.collect import state as collector_state

API_BEATMAPLIST = "https://api.sayobot.cn/beatmaplist"
API_BEATMAPINFO = "https://api.sayobot.cn/v2/beatmapinfo"
URL_DOWNLOAD_MINI = "https://dl.sayobot.cn/beatmaps/download/mini/{sid}"

TYPE_NEW = 2
CLASS_RANKED_APPROVED = 1
MODE_MANIA = 8
OSU_MODE_MANIA = 3

REFERER = os.environ.get("SAYOBOT_REFERER", "https://osu.sayobot.cn/")
_CONTACT = os.environ.get("AUDIO2MAP_CONTACT", "").strip()
USER_AGENT = os.environ.get(
    "SAYOBOT_USER_AGENT",
    "Audio2Map-mania-4k-dataset"
    + (f" (contact: {_CONTACT}; +https://osu.sayobot.cn)" if _CONTACT else ""),
)
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Referer": REFERER}

AUDIO_EXT = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus"}


def _tmp_dir() -> Path:
    return collector_dir() / "_tmp"


def _log_path() -> Path:
    return collector_dir() / "download.log"


def setup_logging(*, truncate_log: bool = True) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(path, mode="w" if truncate_log else "a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def layout_ok(set_dir: Path) -> bool:
    """Root dir: only ``.osu`` + referenced audio, no subdirs or extras."""
    if not set_dir.is_dir() or any(p.is_dir() for p in set_dir.iterdir()):
        return False
    files = [p for p in set_dir.iterdir() if p.is_file()]
    if any(p.suffix.lower() not in AUDIO_EXT | {".osu"} for p in files):
        return False
    osu = [p for p in files if p.suffix.lower() == ".osu"]
    audio = {p.resolve() for p in files if p.suffix.lower() in AUDIO_EXT}
    if not osu or not audio:
        return False
    mapping = chart_audio_map(set_dir, osu)
    if mapping is None:
        return False
    return {p.resolve() for p in mapping.values()} == audio


def flatten(set_dir: Path) -> None:
    for path in sorted(set_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_file() or path.parent == set_dir or path.suffix.lower() not in AUDIO_EXT | {".osu"}:
            continue
        dest = set_dir / path.name
        if dest.exists():
            continue
        path.replace(dest)
    for path in sorted(set_dir.rglob("*"), reverse=True):
        if path.is_dir() and path != set_dir and not any(path.iterdir()):
            path.rmdir()


def _audio_index(set_dir: Path) -> dict[str, Path]:
    return {
        p.name.lower(): p
        for p in set_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXT
    }


def chart_audio_map(set_dir: Path, osu_files: list[Path]) -> dict[Path, Path] | None:
    """Each ``.osu`` → its ``AudioFilename`` file. ``None`` if any chart is unbound."""
    index = _audio_index(set_dir)
    out: dict[Path, Path] = {}
    for osu in osu_files:
        name = audio_filename(osu)
        if not name:
            return None
        found = index.get(Path(name).name.lower())
        if found is None:
            return None
        out[osu] = found
    return out


def _rewrite_audio_filename(osu: Path, audio_name: str) -> None:
    lines = osu.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() != "[General]":
            continue
        for j in range(i + 1, len(lines)):
            s = lines[j].strip()
            if s.startswith("AudioFilename:"):
                lines[j] = f"AudioFilename:{audio_name}\n"
                break
            if s.startswith("[") and s.endswith("]"):
                lines.insert(j, f"AudioFilename:{audio_name}\n")
                break
        break
    osu.write_text("".join(lines), encoding="utf-8")


def prune(set_dir: Path) -> None:
    osu_files = sorted(p for p in set_dir.rglob("*.osu") if p.is_file())
    if not osu_files:
        return
    mapping = chart_audio_map(set_dir, osu_files)
    if mapping is None:
        return
    keep_audio: set[Path] = set()
    for osu, src in mapping.items():
        keep_audio.add(src.resolve())
        name = audio_filename(osu) or ""
        if (osu.parent / name).resolve() != src.resolve():
            _rewrite_audio_filename(osu, src.name)
    keep = keep_audio | {p.resolve() for p in osu_files}
    for p in list(set_dir.rglob("*")):
        if p.is_file() and p.resolve() not in keep:
            p.unlink()
    for p in sorted(set_dir.rglob("*"), reverse=True):
        if p.is_dir() and p != set_dir and not any(p.iterdir()):
            p.rmdir()


def prepare_set(set_dir: Path) -> None:
    flatten(set_dir)
    prune(set_dir)
    flatten(set_dir)


def delete_if_bad(set_dir: Path) -> bool:
    """Normalize layout; delete folder if still invalid. Returns True if deleted."""
    prepare_set(set_dir)
    if layout_ok(set_dir):
        return False
    logging.warning("sid=%s deleted (bad layout)", set_dir.name)
    shutil.rmtree(set_dir, ignore_errors=True)
    return True


def apply_4k_filter(set_dir: Path) -> bool:
    kept = False
    for osu in list(set_dir.rglob("*.osu")):
        if is_mania_4k(osu):
            kept = True
        else:
            osu.unlink()
    return kept


def fetch_page(session: requests.Session, offset: int, limit: int) -> dict[str, Any]:
    resp = session.get(
        API_BEATMAPLIST,
        params={"L": limit, "O": offset, "T": TYPE_NEW, "C": CLASS_RANKED_APPROVED},
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status", 0) != 0:
        raise RuntimeError(payload)
    return payload


def info_has_mania_4k(payload: dict[str, Any]) -> bool | None:
    """True if any diff is mania 4K. False if payload is complete and none are. None if unknown."""
    if payload.get("status", 0) != 0:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    bids = data.get("bid_data")
    if not isinstance(bids, list) or not bids:
        return None
    for bid in bids:
        if not isinstance(bid, dict):
            continue
        try:
            mode = int(bid.get("mode", -1))
            cs = float(bid.get("CS", -1))
        except (TypeError, ValueError):
            continue
        if mode == OSU_MODE_MANIA and round(cs) == 4:
            return True
    return False


def fetch_beatmapinfo(session: requests.Session, sid: int) -> dict[str, Any]:
    resp = session.get(API_BEATMAPINFO, params={"K": sid, "V": sid}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(payload)
    return payload


def sid_has_mania_4k(session: requests.Session, sid: int) -> bool | None:
    try:
        return info_has_mania_4k(fetch_beatmapinfo(session, sid))
    except Exception as exc:
        logging.warning("sid=%s beatmapinfo failed: %s", sid, exc)
        return None


def download_mini(sid: int, dest: Path) -> None:
    """Fetch mini osz with curl (requests/TLS on some GPU hosts does not work here)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = URL_DOWNLOAD_MINI.format(sid=sid)
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required to download beatmaps")

    proc = subprocess.run(
        [
            curl,
            "-fL",
            "--http1.1",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--max-time",
            "300",
            "-A",
            USER_AGENT,
            "-e",
            REFERER,
            "-o",
            str(dest),
            url,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).decode(errors="replace")[-500:].strip()
        raise RuntimeError(f"curl failed sid={sid}: {err or proc.returncode}")
    size = dest.stat().st_size if dest.is_file() else 0
    if size < 1024:
        raise RuntimeError(f"download too small: {size} bytes, sid={sid}")


def _safe_extract(osz: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(osz, "r") as zf:
        for info in zf.infolist():
            out = (dest / info.filename).resolve()
            if not out.is_relative_to(dest):
                raise RuntimeError(f"unsafe zip path: {info.filename}")
        zf.extractall(dest)


def process_sid(sid: int) -> str:
    set_dir = raw_dir() / str(sid)
    if set_dir.is_dir():
        if not delete_if_bad(set_dir) and any(is_mania_4k(p) for p in set_dir.glob("*.osu")):
            return "already_complete"

    osz = _tmp_dir() / f"{sid}.osz"
    try:
        download_mini(sid, osz)
        if set_dir.exists():
            shutil.rmtree(set_dir)
        set_dir.mkdir(parents=True)
        _safe_extract(osz, set_dir)
        if not apply_4k_filter(set_dir):
            shutil.rmtree(set_dir, ignore_errors=True)
            return "skipped_no_4k"
        if delete_if_bad(set_dir):
            return "skipped_no_4k"
        return "completed"
    finally:
        osz.unlink(missing_ok=True)


def _tick(pbar: tqdm, sid: int, status: str) -> None:
    pbar.update(1)
    pbar.set_postfix(sid=sid, st=status, refresh=False)


def handle_sid(sid: int, pbar: tqdm) -> tuple[str, str | None]:
    try:
        status = process_sid(sid)
        err = None
    except Exception as exc:
        logging.exception("sid=%s failed: %s", sid, exc)
        shutil.rmtree(raw_dir() / str(sid), ignore_errors=True)
        status = "failed"
        err = str(exc)
    n = len(list((raw_dir() / str(sid)).glob("*.osu"))) if status == "completed" else 0
    logging.info("sid=%s | %s | osu=%d", sid, status, n)
    _tick(pbar, sid, status)
    return status, err


def run(
    *,
    target: int | None = None,
    page_size: int = 50,
    list_delay: float = 1.0,
    max_pages: int | None = None,
    status_only: bool = False,
) -> None:
    setup_logging(truncate_log=not status_only)
    raw_dir().mkdir(parents=True, exist_ok=True)
    _tmp_dir().mkdir(parents=True, exist_ok=True)
    st = collector_state.load_state()

    if status_only:
        collector_state.print_state_stats(st)
        return

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    offset = int(st.get("offset", 0))
    processed = pages = 0
    with tqdm(total=target, desc="collect", unit="set", dynamic_ncols=True) as pbar:
        while True:
            if max_pages is not None and pages >= max_pages:
                break
            if target is not None and processed >= target:
                break
            payload = fetch_page(session, offset, page_size)
            pages += 1
            entries = payload.get("data") or []
            if not entries:
                collector_state.persist(st, offset)
                break
            for item in entries:
                if target is not None and processed >= target:
                    break
                sid = int(item["sid"])
                if collector_state.should_skip(sid, st):
                    continue
                processed += 1
                modes = int(item.get("modes", 0))
                approved = int(item.get("approved", 0))
                if (modes & MODE_MANIA) == 0:
                    collector_state.apply_result(sid, "skipped_no_mania", st)
                    _tick(pbar, sid, "skip_api")
                elif approved not in (1, 2):
                    collector_state.apply_result(sid, "skipped_class", st)
                    _tick(pbar, sid, "skip_class")
                elif sid_has_mania_4k(session, sid) is False:
                    collector_state.apply_result(sid, "skipped_no_4k", st)
                    _tick(pbar, sid, "skip_info")
                else:
                    result, err = handle_sid(sid, pbar)
                    collector_state.apply_result(sid, result, st, err)
            endid = int(payload.get("endid", 0))
            done = (target is not None and processed >= target) or endid == 0
            if not done:
                offset = endid
            collector_state.persist(st, offset)
            if done or (max_pages is not None and pages >= max_pages):
                break
            if list_delay > 0:
                time.sleep(list_delay)
    collector_state.print_state_stats(st)
