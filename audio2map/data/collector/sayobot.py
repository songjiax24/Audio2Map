"""Batch download osu! mania 4K beatmaps from Sayobot."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from audio2map.data.collector import state as collector_state
from audio2map.utils.paths import collector_dir, raw_dir

API_BEATMAPLIST = "https://api.sayobot.cn/beatmaplist"
URL_DOWNLOAD_MINI = "https://dl.sayobot.cn/beatmaps/download/mini/{sid}"

TYPE_NEW = 2
CLASS_RANKED_APPROVED = 1
MODE_MANIA = 8

REFERER = os.environ.get("SAYOBOT_REFERER", "https://osu.sayobot.cn/")
USER_AGENT = os.environ.get(
    "SAYOBOT_USER_AGENT",
    "Audio2Map-mania-4k-dataset (contact: Damocleser_L@163.com; +https://osu.sayobot.cn)",
)
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Referer": REFERER}

AUDIO_EXT = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus"}
_SFX_STEM = re.compile(
    r"hit(?:normal|whistle|finish|clap)|slider|^(soft|drum|normal|taiko)-hit|"
    r"combobreak|pause",
    re.I,
)

DATASET_DIR = raw_dir()
TMP_DIR = collector_dir() / "_tmp"
LOG_PATH = collector_dir() / "download.log"


def setup_logging(*, truncate_log: bool = True) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w" if truncate_log else "a", encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _parse_osu(osu_path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    cur: str | None = None
    for raw in osu_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            cur = line
            sections[cur] = {}
        elif cur and ":" in line:
            k, _, v = line.partition(":")
            sections[cur][k.strip()] = v.strip()
    return sections


def is_mania_4k(osu_path: Path) -> bool:
    sec = _parse_osu(osu_path)
    g, d = sec.get("[General]", {}), sec.get("[Difficulty]", {})
    try:
        if int(g.get("Mode", -1)) != 3:
            return False
        cs = float(d["CircleSize"])
    except (KeyError, ValueError):
        return False
    return abs(cs - 4.0) < 1e-6 or round(cs) == 4


def audio_filename(osu_path: Path) -> str | None:
    return _parse_osu(osu_path).get("[General]", {}).get("AudioFilename")


def is_sfx(path: Path) -> bool:
    if path.suffix.lower() not in AUDIO_EXT:
        return False
    if path.stat().st_size < 200_000:
        return True
    return bool(_SFX_STEM.search(path.stem))


def layout_ok(set_dir: Path) -> bool:
    """Root dir: exactly 1 mp3 + >=1 osu, no subdirs or extra files."""
    if not set_dir.is_dir() or any(p.is_dir() for p in set_dir.iterdir()):
        return False
    files = [p for p in set_dir.iterdir() if p.is_file()]
    mp3 = [p for p in files if p.suffix.lower() == ".mp3"]
    osu = [p for p in files if p.suffix.lower() == ".osu"]
    return len(mp3) == 1 and osu and len(files) == len(mp3) + len(osu)


def flatten(set_dir: Path) -> None:
    move = {".osu", ".mp3", *AUDIO_EXT}
    for path in sorted(set_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_file() or path.parent == set_dir or path.suffix.lower() not in move:
            continue
        dest = set_dir / path.name
        if dest.exists() and dest.resolve() != path.resolve():
            dest = set_dir / f"_{path.parent.name}_{path.name}"
        path.replace(dest)
    for path in sorted(set_dir.rglob("*"), reverse=True):
        if path.is_dir() and path != set_dir and not any(path.iterdir()):
            path.rmdir()


def find_audio(set_dir: Path, osu_files: list[Path]) -> Path | None:
    pool = [p for p in set_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT and not is_sfx(p)]
    if not pool:
        return None
    for name, _ in Counter(n for o in osu_files if (n := audio_filename(o))).most_common():
        for p in pool:
            if p.name.lower() == Path(name).name.lower():
                return p
    return max(pool, key=lambda p: p.stat().st_size)


def prune(set_dir: Path) -> None:
    osu_files = sorted(set_dir.rglob("*.osu"))
    if not osu_files:
        return
    main = find_audio(set_dir, osu_files)
    if main is None:
        return
    mp3 = set_dir / f"{main.stem}.mp3"
    if main.suffix.lower() == ".mp3":
        if main.resolve() != mp3.resolve():
            mp3.unlink(missing_ok=True)
            main.replace(mp3)
    else:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required for non-mp3 audio")
        subprocess.run(
            [ffmpeg, "-y", "-i", str(main), "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", str(mp3)],
            check=True,
            capture_output=True,
        )
        main.unlink(missing_ok=True)
    for p in list(set_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT and p.resolve() != mp3.resolve():
            p.unlink()
    for osu in osu_files:
        if not osu.exists():
            continue
        lines = osu.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.strip() == "[General]":
                for j in range(i + 1, len(lines)):
                    s = lines[j].strip()
                    if s.startswith("AudioFilename:"):
                        lines[j] = f"AudioFilename:{mp3.name}\n"
                        break
                    if s.startswith("[") and s.endswith("]"):
                        lines.insert(j, f"AudioFilename:{mp3.name}\n")
                        break
                break
        osu.write_text("".join(lines), encoding="utf-8")
    for p in list(set_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() not in {".osu", ".mp3"}:
            p.unlink()
    for p in sorted(set_dir.rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
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


def download_mini(session: requests.Session, sid: int, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = URL_DOWNLOAD_MINI.format(sid=sid)

    cmd = [
        "curl",
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
    ]

    subprocess.run(cmd, check=True)

    size = dest.stat().st_size
    if size < 1024:
        raise RuntimeError(f"download too small: {size} bytes, sid={sid}")

    return size


def process_sid(session: requests.Session, sid: int, delay: float) -> str:
    set_dir = DATASET_DIR / str(sid)
    if set_dir.is_dir():
        if not delete_if_bad(set_dir) and any(is_mania_4k(p) for p in set_dir.glob("*.osu")):
            return "already_complete"

    osz = TMP_DIR / f"{sid}.osz"
    try:
        size = download_mini(session, sid, osz)
        if set_dir.exists():
            shutil.rmtree(set_dir)
        set_dir.mkdir(parents=True)
        with zipfile.ZipFile(osz, "r") as zf:
            zf.extractall(set_dir)
        if not apply_4k_filter(set_dir):
            shutil.rmtree(set_dir, ignore_errors=True)
            return "skipped_no_4k"
        if delete_if_bad(set_dir):
            return "skipped_no_4k"
        time.sleep(min(max(delay, size / (1024 * 1024) / 5.0), 30.0))
        return "completed"
    finally:
        osz.unlink(missing_ok=True)


def handle_sid(session: requests.Session, sid: int, delay: float, pbar: tqdm) -> str:
    try:
        status = process_sid(session, sid, delay)
    except Exception as exc:
        logging.exception("sid=%s failed: %s", sid, exc)
        shutil.rmtree(DATASET_DIR / str(sid), ignore_errors=True)
        status = "failed"
    n = len(list((DATASET_DIR / str(sid)).glob("*.osu"))) if status == "completed" else 0
    logging.info(
        "%s | sid=%s | %s | osu=%d",
        datetime.now().isoformat(timespec="seconds"),
        sid,
        status,
        n,
    )
    pbar.update(1)
    pbar.set_postfix(sid=sid, st=status, refresh=False)
    return status


def run(
    *,
    target: int | None = None,
    page_size: int = 50,
    list_delay: float = 1.0,
    download_delay: float = 2.0,
    max_pages: int | None = None,
    status_only: bool = False,
) -> None:
    setup_logging(truncate_log=not status_only)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
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
            time.sleep(list_delay)
            pages += 1
            entries = payload.get("data") or []
            if not entries:
                offset = 0
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
                    pbar.update(1)
                    pbar.set_postfix(sid=sid, st="skip_api", refresh=False)
                elif approved not in (1, 2):
                    pbar.update(1)
                    pbar.set_postfix(sid=sid, st="skip_class", refresh=False)
                else:
                    result = handle_sid(session, sid, download_delay, pbar)
                    err = "download failed" if result == "failed" else None
                    collector_state.apply_result(sid, result, st, err)
                collector_state.persist(st, offset)
            if target is not None and processed >= target:
                break
            endid = int(payload.get("endid", 0))
            if endid == 0:
                offset = 0
                collector_state.persist(st, offset)
                break
            offset = endid
            collector_state.persist(st, offset)
    collector_state.print_state_stats(st)
