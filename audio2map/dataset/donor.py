"""Persistent target → donor chart mapping for ``random_full``.

The file is a dataset contract next to the frozen split. The seed is recorded
for audit. Lookup is by raw-relative path, so a later condition schema cannot
retarget the same index onto a different chart.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from audio2map.dataset.split import SplitError, chart_relpath
from audio2map.utils.paths import splits_dir


class DonorMapError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DonorMap:
    seed: int
    pairs: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "pairs": [
                {"target": target, "donor": self.pairs[target]}
                for target in sorted(self.pairs)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> DonorMap:
        pairs: dict[str, str] = {}
        for row in data.get("pairs", []):
            target = str(row["target"])
            donor = str(row["donor"])
            if target in pairs:
                raise DonorMapError(f"duplicate target in donor map: {target}")
            pairs[target] = donor
        return cls(seed=int(data["seed"]), pairs=pairs)


def default_donor_map_path() -> Path:
    return splits_dir() / "random_full.json"


def candidate_universe_id(relpaths: list[str]) -> str:
    """Identity of a candidate set. Order does not change the id."""
    blob = "\n".join(sorted(set(relpaths)))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def donor_pool_report(mapping: DonorMap, targets: list[str]) -> dict:
    """Audit view: each target, its donor, and the candidate-set identity."""
    rels = sorted(set(targets))
    return {
        "candidate_count": len(rels),
        "universe_id": candidate_universe_id(rels),
        "pairs": [{"target": rel, "donor": mapping.pairs[rel]} for rel in rels],
    }


def chart_key(chart_path: Path, raw_root: Path) -> str:
    try:
        return chart_relpath(chart_path, raw_root)
    except ValueError as exc:
        raise DonorMapError(str(exc)) from exc
    except SplitError as exc:
        raise DonorMapError(str(exc)) from exc


def save_donor_map(mapping: DonorMap, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise DonorMapError(
            f"donor map already exists: {path}; write a new file instead of overwriting"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping.to_dict(), indent=2), encoding="utf-8")


def load_donor_map(path: Path) -> DonorMap:
    path = Path(path)
    if not path.is_file():
        raise DonorMapError(f"donor map missing: {path}")
    return DonorMap.from_dict(json.loads(path.read_text(encoding="utf-8")))


def create_donor_map(targets: list[str], *, seed: int) -> DonorMap:
    """Draw one donor path per target from ``targets`` itself.

    Self-draws are allowed. ``targets`` is sorted before sampling so the draw
    does not depend on caller order. The seed is stored and is not the identity
    used on later loads.
    """
    pool = sorted(set(targets))
    if not pool:
        raise DonorMapError("cannot create a donor map from an empty chart pool")
    rng = random.Random(seed)
    return DonorMap(seed=seed, pairs={target: rng.choice(pool) for target in pool})


def load_or_create_donor_map(
    path: Path,
    targets: list[str],
    *,
    seed: int,
) -> DonorMap:
    """Load an existing map, or create it once when ``path`` is absent.

    Every target in this run must already be a key. Missing keys fail. Extra
    keys already in the file are kept and are not rewritten.
    """
    path = Path(path)
    if path.exists():
        mapping = load_donor_map(path)
        missing = sorted(set(targets) - set(mapping.pairs))
        if missing:
            raise DonorMapError(
                f"{len(missing)} target chart(s) missing from donor map {path}: {missing[:20]}"
            )
        return mapping
    mapping = create_donor_map(targets, seed=seed)
    save_donor_map(mapping, path)
    return mapping
