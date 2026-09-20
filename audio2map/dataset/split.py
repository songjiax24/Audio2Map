"""Beatmapset-level train/val/test split from dataset layout.

``group_id`` is the directory of a chart relative to ``raw/``. ``groups`` is the
only assignment source of truth; ``charts`` is a frozen chart-set snapshot
(relative paths), not a preprocessing freeze of audio / grid / chart_meta.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from audio2map.osu.parser import parse_chart_metadata
from audio2map.utils.paths import raw_dir, splits_dir

logger = logging.getLogger(__name__)

SPLIT_SCHEME = "beatmapset_v1"
GROUP_BY = "raw_parent"
DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SplitName = Literal["train", "val", "test"]


class SplitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChartSnapshot:
    relpath: str
    group_id: str
    split: str
    beatmap_id: int | None = None
    beatmap_set_id: int | None = None


@dataclass(frozen=True, slots=True)
class SetIdMismatch:
    relpath: str
    group_id: str
    folder_name: str
    beatmap_set_id: int


@dataclass
class SplitManifest:
    split_scheme: str
    group_by: str
    seed: int
    ratios: dict[str, float]
    groups: dict[str, str]
    charts: list[ChartSnapshot] = field(default_factory=list)
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "split_scheme": self.split_scheme,
            "group_by": self.group_by,
            "seed": self.seed,
            "ratios": dict(self.ratios),
            "groups": dict(self.groups),
            "charts": [asdict(c) for c in self.charts],
            "audit": self.audit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SplitManifest:
        charts = [ChartSnapshot(**row) for row in data.get("charts", [])]
        return cls(
            split_scheme=data["split_scheme"],
            group_by=data["group_by"],
            seed=int(data["seed"]),
            ratios={k: float(v) for k, v in data["ratios"].items()},
            groups={str(k): str(v) for k, v in data["groups"].items()},
            charts=charts,
            audit=dict(data.get("audit") or {}),
        )


def default_split_manifest_path() -> Path:
    return splits_dir() / "beatmapset_v1.json"


def group_id_for_chart(chart_path: Path, raw_root: Path) -> str:
    """Return the leakage group for ``chart_path`` from dataset layout."""
    chart = Path(chart_path).resolve()
    root = Path(raw_root).resolve()
    try:
        rel = chart.relative_to(root)
    except ValueError as exc:
        raise SplitError(f"chart is not under raw root {root}: {chart}") from exc
    parent = rel.parent.as_posix()
    if parent in ("", "."):
        raise SplitError(f"chart is not under a group directory: {chart}")
    return parent


def list_raw_osu_paths(raw_root: Path | None = None) -> list[Path]:
    root = Path(raw_root) if raw_root is not None else raw_dir()
    return sorted(p for p in root.rglob("*.osu") if p.is_file())


def _unit_interval(group_id: str, *, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def _assign_split(group_id: str, *, seed: int, ratios: dict[str, float]) -> SplitName:
    train_r = ratios["train"]
    val_r = ratios["val"]
    u = _unit_interval(group_id, seed=seed)
    if u < train_r:
        return "train"
    if u < train_r + val_r:
        return "val"
    return "test"


def _check_ratios(ratios: dict[str, float]) -> dict[str, float]:
    needed = ("train", "val", "test")
    if any(k not in ratios for k in needed):
        raise SplitError(f"ratios must include {needed}, got {sorted(ratios)}")
    total = sum(float(ratios[k]) for k in needed)
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"ratios must sum to 1, got {total}")
    if any(float(ratios[k]) < 0 for k in needed):
        raise SplitError("ratios must be non-negative")
    return {k: float(ratios[k]) for k in needed}


def _folder_name(group_id: str) -> str:
    return Path(group_id).name


def _collect_set_id_mismatches(
    paths: list[Path],
    *,
    raw_root: Path,
) -> list[SetIdMismatch]:
    mismatches: list[SetIdMismatch] = []
    for path in paths:
        group_id = group_id_for_chart(path, raw_root)
        try:
            meta = parse_chart_metadata(path)
        except (OSError, ValueError) as exc:
            logger.warning("skip BeatmapSetID check for %s: %s", path, exc)
            continue
        set_id = meta.beatmap_set_id
        if set_id is None or set_id <= 0:
            continue
        folder = _folder_name(group_id)
        if str(set_id) != folder:
            mismatches.append(
                SetIdMismatch(
                    relpath=chart_relpath(path, raw_root),
                    group_id=group_id,
                    folder_name=folder,
                    beatmap_set_id=set_id,
                )
            )
    return mismatches


def build_split_manifest(
    *,
    raw_root: Path | None = None,
    seed: int = 0,
    ratios: dict[str, float] | None = None,
    allow_set_id_mismatch: bool = False,
) -> SplitManifest:
    """Group every ``raw/**/*.osu`` by parent directory and hash-assign splits."""
    raw_root = Path(raw_root).resolve() if raw_root is not None else raw_dir()
    ratios = _check_ratios(ratios or DEFAULT_RATIOS)
    paths = list_raw_osu_paths(raw_root)
    if not paths:
        raise SplitError(f"no .osu files under {raw_root}")

    mismatches = _collect_set_id_mismatches(paths, raw_root=raw_root)
    if mismatches and not allow_set_id_mismatch:
        lines = ", ".join(
            f"{m.relpath} folder={m.folder_name!r} BeatmapSetID={m.beatmap_set_id}"
            for m in mismatches[:20]
        )
        extra = "" if len(mismatches) <= 20 else f" (+{len(mismatches) - 20} more)"
        raise SplitError(
            f"{len(mismatches)} BeatmapSetID/folder conflicts (pass "
            f"--allow-set-id-mismatch to override): {lines}{extra}"
        )

    groups: dict[str, str] = {}
    charts: list[ChartSnapshot] = []
    for path in paths:
        group_id = group_id_for_chart(path, raw_root)
        split = groups.get(group_id)
        if split is None:
            split = _assign_split(group_id, seed=seed, ratios=ratios)
            groups[group_id] = split
        try:
            meta = parse_chart_metadata(path)
            beatmap_id = meta.beatmap_id
            beatmap_set_id = meta.beatmap_set_id
        except (OSError, ValueError):
            beatmap_id = None
            beatmap_set_id = None
        charts.append(
            ChartSnapshot(
                relpath=chart_relpath(path, raw_root),
                group_id=group_id,
                split=split,
                beatmap_id=beatmap_id,
                beatmap_set_id=beatmap_set_id,
            )
        )

    audit = {
        "n_charts": len(charts),
        "n_groups": len(groups),
        "set_id_mismatches": [asdict(m) for m in mismatches],
    }
    return SplitManifest(
        split_scheme=SPLIT_SCHEME,
        group_by=GROUP_BY,
        seed=seed,
        ratios=ratios,
        groups=groups,
        charts=charts,
        audit=audit,
    )


def chart_relpath(chart_path: Path, raw_root: Path) -> str:
    return Path(chart_path).resolve().relative_to(Path(raw_root).resolve()).as_posix()


def split_group_chart_counts(manifest: SplitManifest) -> dict[str, tuple[int, int]]:
    """Return ``{split: (n_groups, n_charts)}`` from the frozen split / chart set."""
    n_groups = {"train": 0, "val": 0, "test": 0}
    n_charts = {"train": 0, "val": 0, "test": 0}
    for split in manifest.groups.values():
        if split in n_groups:
            n_groups[split] += 1
    for chart in manifest.charts:
        if chart.split in n_charts:
            n_charts[chart.split] += 1
    return {name: (n_groups[name], n_charts[name]) for name in ("train", "val", "test")}


def manifest_sha256(manifest: SplitManifest) -> str:
    """Stable hash of split scheme, group assignment, and chart-set identity.

    Ignores JSON formatting, ``charts`` order, ``audit``, and per-chart metadata
    ids. Changing a group split or the relative-path set changes the digest.
    """
    ratios = _check_ratios(manifest.ratios)
    payload = {
        "split_scheme": manifest.split_scheme,
        "group_by": manifest.group_by,
        "seed": int(manifest.seed),
        "ratios": {k: ratios[k] for k in ("test", "train", "val")},
        "groups": [[gid, manifest.groups[gid]] for gid in sorted(manifest.groups)],
        "charts": [
            [c.relpath, c.group_id, c.split]
            for c in sorted(manifest.charts, key=lambda row: row.relpath)
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_chart_set(
    manifest: SplitManifest,
    disk_paths: list[Path],
    *,
    raw_root: Path,
) -> None:
    """Fail-fast if the on-disk ``.osu`` set or groups/charts snapshot drifted."""
    raw_root = Path(raw_root).resolve()
    snapshot_groups: set[str] = set()
    for chart in manifest.charts:
        snapshot_groups.add(chart.group_id)
        if chart.group_id not in manifest.groups:
            raise SplitError(
                f"charts[] group_id {chart.group_id!r} ({chart.relpath}) is not in groups[]"
            )
        assigned = manifest.groups[chart.group_id]
        if chart.split != assigned:
            raise SplitError(
                f"charts[] split for {chart.relpath} is {chart.split!r} but "
                f"groups[{chart.group_id!r}] is {assigned!r}"
            )
    extra_group_keys = sorted(set(manifest.groups) - snapshot_groups)
    if extra_group_keys:
        raise SplitError(
            f"groups[] has keys with no charts snapshot: {extra_group_keys[:20]}"
            + (f" (+{len(extra_group_keys) - 20} more)" if len(extra_group_keys) > 20 else "")
        )

    disk_rel = {chart_relpath(path, raw_root) for path in disk_paths}
    snapshot_rel = {chart.relpath for chart in manifest.charts}
    extra = sorted(disk_rel - snapshot_rel)
    missing = sorted(snapshot_rel - disk_rel)
    if extra:
        raise SplitError(
            f"{len(extra)} chart(s) on disk are not in the split manifest: {extra[:20]}"
            + (f" (+{len(extra) - 20} more)" if len(extra) > 20 else "")
            + "; write a new manifest (e.g. beatmapset_v2.json) instead of editing this one"
        )
    if missing:
        raise SplitError(
            f"{len(missing)} chart(s) in the split manifest are missing from disk: {missing[:20]}"
            + (f" (+{len(missing) - 20} more)" if len(missing) > 20 else "")
        )
    by_rel = {chart.relpath: chart for chart in manifest.charts}
    for path in disk_paths:
        rel = chart_relpath(path, raw_root)
        layout_group = group_id_for_chart(path, raw_root)
        snap = by_rel[rel]
        if snap.group_id != layout_group:
            raise SplitError(
                f"charts[] group_id {snap.group_id!r} for {rel} != layout {layout_group!r}"
            )


def save_split_manifest(manifest: SplitManifest, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise SplitError(
            f"split manifest already exists: {path}; write a new file "
            f"(e.g. splits/beatmapset_v2.json) instead of overwriting"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def load_split_manifest(
    path: Path,
    *,
    expected_scheme: str = SPLIT_SCHEME,
    expected_group_by: str = GROUP_BY,
    expected_seed: int | None = None,
    expected_ratios: dict[str, float] | None = None,
) -> SplitManifest:
    path = Path(path)
    if not path.is_file():
        raise SplitError(f"split manifest missing: {path} (pass --init-split to create)")
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = SplitManifest.from_dict(data)
    if manifest.split_scheme != expected_scheme:
        raise SplitError(
            f"split_scheme {manifest.split_scheme!r} != {expected_scheme!r} in {path}"
        )
    if manifest.group_by != expected_group_by:
        raise SplitError(f"group_by {manifest.group_by!r} != {expected_group_by!r} in {path}")
    if expected_seed is not None and manifest.seed != expected_seed:
        raise SplitError(f"split seed {manifest.seed} != {expected_seed} in {path}")
    if expected_ratios is not None:
        got = _check_ratios(manifest.ratios)
        want = _check_ratios(expected_ratios)
        if any(abs(got[k] - want[k]) > 1e-9 for k in ("train", "val", "test")):
            raise SplitError(f"split ratios {got} != {want} in {path}")
    return manifest


def split_paths(
    paths: list[Path],
    manifest: SplitManifest,
    *,
    raw_root: Path | None = None,
) -> dict[str, list[Path]]:
    """Assign current paths using ``manifest.groups`` after chart-set validation."""
    raw_root = Path(raw_root).resolve() if raw_root is not None else raw_dir()
    validate_chart_set(manifest, paths, raw_root=raw_root)
    out: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for path in paths:
        group_id = group_id_for_chart(path, raw_root)
        split = manifest.groups[group_id]
        out[split].append(path)
    return out
