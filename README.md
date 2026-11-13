# Audio2Map

Audio2Map generates **osu!mania 4K** charts from audio, beat timing, and
difficulty and pattern conditions. Several charts can interpret the same song
and conditions. Evaluation uses reference matching, condition response,
diversity, and structural checks to measure different model capabilities.

| Version | Training and generation | Commands |
|---------|-------------------------|----------|
| v1 | Whole-chart training, frame-level audio with beat phase, red-line timing, and 18 masked condition fields | `audio2map-train-v1`, `audio2map-infer-v1`, `audio2map-eval-v1` |
| v0 | Constant BPM, meter 4, 16-bar training windows, and overlapping generation | `audio2map-train`, `audio2map-infer`, `audio2map-eval` |

The two versions use different audio features, token vocabularies, and
checkpoints. The [web demo](demo/README.md) uses v0.

## Setup

Use Python 3.12, [uv](https://docs.astral.sh/uv/), and FFmpeg with `ffprobe` on
`PATH`. On Debian/Ubuntu, install `ffmpeg` and `libsndfile1`. The v1 training
corpus was prepared with FFmpeg 7.1.1. See the
[v1 audio specification](docs/v1_design.md) for decoder and resampler details.

```bash
uv sync --python 3.12 --extra train --extra eval --extra dev
source .venv/bin/activate
export AUDIO2MAP_DATA_ROOT=/path/to/audio2map_data
```

The `train` extra provides PyTorch for training, inference, and model evaluation.
The `eval` extra adds Matplotlib for HTML figures. The `dev` extra provides tests
and linting. For CUDA runs, verify that PyTorch sees the GPU:

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

The data root defaults to the ignored `.local-data/` directory. Keep raw charts
with their referenced audio under `raw/<set_id>/`. Split manifests go under
`splits/`, and checkpoints, caches, and evaluation runs go under `processed/`.
Keep weights, chart/audio assets, local manifests, and `exports/` out of Git.

## v1 workflow

### Train

Put the existing `beatmapset_v1_drop2509435.json` manifest under `splits/`, or
supply your own compatible manifest with `--split-manifest`. The trainer builds
v1 caches from raw charts and audio. The default config uses CUDA and bf16 and
saves `step_N.pt` files under `processed/checkpoints/v1_chart/`.

```bash
audio2map-train-v1 --config configs/train/train_v1.yaml
```

Pass `--resume /path/to/step_N.pt` to continue a run. `--max-steps` is the total
step count.

### Generate a chart

Provide the audio file, a timing file with a `[TimingPoints]` section, and a
ChartMeta JSON object. Inference reads red lines from the timing file. Save
`{"cond_vec_version": 3, "official_sr": 5.0}` as `condition.json` for a
minimal request. Omitted condition fields are masked. The condition fields
and their units are documented in the [v1 design](docs/v1_design.md).

```bash
checkpoint="$AUDIO2MAP_DATA_ROOT/processed/checkpoints/v1_chart/step_200000.pt"
audio2map-infer-v1 --checkpoint "$checkpoint" \
  --audio path/to/song.mp3 --timing path/to/timing.osu \
  --cond condition.json --out path/to/generated.osu
```

Replace the step number with an available checkpoint. The generated chart
references the audio by filename. For an `.osz` package, zip the chart and its
referenced audio together and give the archive the `.osz` extension.

### Evaluate a checkpoint

Start with a small validation subset using teacher forcing. It needs no fitted
motif vocabulary.

```bash
audio2map-eval-v1 --mode teacher --split val --limit 4 \
  --checkpoint "$checkpoint" \
  --out "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/teacher_smoke"
```

For constrained generation metrics, fit the motif vocabulary on the training
split first. This example uses one seed and a small validation pool:

```bash
eval_root="$AUDIO2MAP_DATA_ROOT/processed/eval/v1"
audio2map-eval-v1 --mode fit-motifs --split train --out "$eval_root/motif_train"
audio2map-eval-v1 --config configs/eval/v1.yaml --mode constrained \
  --split val --limit 4 --seeds 0 --checkpoint "$checkpoint" \
  --motif-vocab "$eval_root/motif_train/motifs.json" \
  --out "$eval_root/constrained_smoke" --html
```

`records.jsonl` holds the per-chart results and `summary.json` holds aggregate
metrics. `summary.md` shows the main tables. With `--html`, open `index.html`
for figures and tables. Use a separate output directory for each checkpoint
and configuration. See the [evaluation guide](docs/v1_eval.md) for a full run,
checkpoint comparisons, rescoring, and audio-shift and condition probes.

## v0 workflow

v0 uses cached tick-level audio grids and
`splits/beatmapset_v1.json`. For a new v0 data disk, add `--init-split` to the
first training command. If raw charts are missing, see the
[data collector guide](docs/data_collection.md) for the Sayobot collection
pipeline.

```bash
audio2map-grid audit
audio2map-grid precompute
audio2map-train --config configs/train/train_multi.yaml
```

The v0 training config writes to `processed/checkpoints/formal_enc_dec_v3/`.
To run teacher evaluation and generate with that checkpoint:

```bash
checkpoint="$AUDIO2MAP_DATA_ROOT/processed/checkpoints/formal_enc_dec_v3/step_200000.pt"
audio2map-eval --mode teacher --checkpoint "$checkpoint" --split test \
  --split-manifest "$AUDIO2MAP_DATA_ROOT/splits/beatmapset_v1.json"
audio2map-infer --checkpoint "$checkpoint" \
  --osu path/to/template.osu --out path/to/generated.osu
```

`audio2map-meta` builds the legacy `chart_meta/manifest.jsonl` used by the demo.
See the [v0 token reference](docs/chart_tokens.md) for the 821-token vocabulary
and window representation.

## Repository guide

| Path | Contents |
|------|----------|
| `audio2map/` | Parsing, grids, tokens, features, models, training, generation, evaluation, and CLI entry points |
| `configs/` | [YAML configuration and overrides](configs/README.md) |
| `tools/` and `scripts/` | Diagnostics and data pipelines |
| `tests/` | Tests that mirror the package |
| `demo/` | [v0 web demo](demo/README.md) |

[Architecture](docs/ARCHITECTURE.md) describes package boundaries and ownership.
The [v1 design](docs/v1_design.md) records the model and data contracts.
For diagnostic methods and artifacts, see
[audio alignment](docs/audio_alignment.md), [onset probes](docs/onset_probe.md),
and [generation logits](docs/generation_logits.md).

## Tests

```bash
pytest
lint-imports
```

Data-backed tests skip when the required charts or frozen artifacts are absent.

## License

Code by the project authors. Beatmap assets belong to their respective creators
and rights holders.
