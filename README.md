# Audio2Map

Generate **osu!mania 4K** charts from audio and global conditions (deep learning).

> **Resuming work / new instance:** read **[docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md)** first (paths, checkpoints, precompute disk issue, commands).  
> **Authoritative v2 spec:** [docs/v2_spec.md](docs/v2_spec.md)

## Layout

```
/root/audio2map/              # code (this repo)
/root/autodl-tmp/audio2map_data/
├── raw/                      # downloaded sets: {sid}/audio.mp3 + *.osu
├── processed/                # OLD MVP (10ms mel+events) — archive
├── processed_v2/             # v2: audio_grid/, checkpoints/, logs/
├── chart_meta/               # per-chart SR/MSD/pattern manifest.jsonl
└── .collector/               # state.json, download.log, _tmp/
```

Override data root: `export AUDIO2MAP_DATA_ROOT=/path/to/data`

## Setup

**New instance (5090, data disk only):** see [docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md) §0–§3.

### Option A — uv (recommended on fresh machine)

```bash
apt-get install -y ffmpeg libsndfile1   # if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /root/audio2map
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,train]"
# Install torch for your CUDA / 5090 from https://pytorch.org/get-started/locally/
export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data
```

### Option B — conda

```bash
cd /root/audio2map
conda env create -f environment.yml
conda activate audio2map
export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data
```

Requires **ffmpeg** (conda env includes it). Reinstall **torch** on 5090 if CUDA arch mismatch.

## Collect data

```bash
# download (resumes from .collector/state.json)
python scripts/collect.py --target 1000 --page-size 50 --list-delay 1 --download-delay 2

# or after pip install -e .
audio2map-collect --target 1000

# show progress
python scripts/collect.py --status
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--target N` | (none) | Process **N list entries** this run (includes skips) |
| `--page-size` | 50 | List API page size |
| `--list-delay` | 1.0 | Seconds after each list request |
| `--download-delay` | 2.0 | Min seconds after each download (+ size-based backoff) |
| `--max-pages` | (none) | Stop after N list pages |
| `--status` | — | Print stats and exit |

See [docs/data_collection.md](docs/data_collection.md) for Sayobot API details.

## v2 train / eval (current)

```bash
conda activate audio2map
export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data

python scripts/train_v2.py --require-grid --steps 3000 --batch-size 8 --samples-per-chart 2
python scripts/eval_v2.py --mode teacher --checkpoint processed_v2/checkpoints/trial_multi/step_3000.pt --osu "..." --start-bar -1
python scripts/precompute_audio_grid.py   # CPU; needs ~80GB+ data disk for full run
```

See [docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md) for checkpoint paths and known issues.

## Roadmap

- [x] Project skeleton + Sayobot collector
- [x] osu! parser + v2 ROW token format
- [x] v2 pipeline: cond_vec, audio_grid, window dataset
- [x] AudioChartModel + debug overfit + multi-chart trial train
- [x] Constrained decode + overlap inference + `.osu` export + eval
- [ ] Full precompute (blocked by 50G data disk on last instance)
- [ ] Production-quality multi-chart training + generate eval

### Validate parser + frame grid

```bash
PYTHONPATH=. python scripts/validate_osu.py
PYTHONPATH=. python scripts/analyze_timing.py
```

See [docs/v2_spec.md](docs/v2_spec.md) for v2 ROW tokenization. Legacy 10ms format: [docs/chart_tokens.md](docs/chart_tokens.md).

### Preprocess

```bash
conda activate audio2map
python scripts/preprocess.py --limit 10   # smoke test
python scripts/preprocess.py            # all sets (~4840)
```

Output: `$AUDIO2MAP_DATA_ROOT/processed/{sid}.npz`, `{sid}.meta.json`, `manifest.jsonl`.
Each set uses the **highest OD** mania 4K chart; mel and chart frames share `hop_ms=10`.

## License

Code: project authors. Beatmap assets: respective creators / osu! rights holders.
