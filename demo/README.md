# Audio2Map Web Demo

Browser demo for generating osu!mania 4K charts from uploaded audio. Weights and the condition manifest are **local runtime assets** (gitignored); place them yourself or point env vars at the data disk.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python env + packages)
- Node.js 18+ (frontend) — install system-wide; do not commit `node_modules/` or `demo/.tools/`

## Local assets (required for generate)

| Asset | Default lookup order | Env override |
|-------|----------------------|--------------|
| Checkpoint | `$AUDIO2MAP_DATA_ROOT/processed/checkpoints/formal_enc_dec_v3/step_200000.pt` (else latest `step_*.pt` there, else `<repo>/checkpoint/step_200000.pt`) | `AUDIO2MAP_CHECKPOINT` |
| Condition manifest | `$AUDIO2MAP_DATA_ROOT/chart_meta/manifest.jsonl`, else `<repo>/chart_meta/manifest.jsonl` | `AUDIO2MAP_CHART_META_MANIFEST` |

| Variable | Default |
|----------|---------|
| `AUDIO2MAP_DEVICE` | `cuda` (use `cpu` to force CPU even if a GPU is present) |

Run commands from the **repository root**.

## Setup (uv)

```bash
# one-time: create venv + install project + demo + training deps
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[train,demo]"

# PyTorch CUDA wheel (optional, recommended for inference speed)
# uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

System libraries (Debian/Ubuntu), mainly for librosa to decode mp3:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
```

## Start backend

With an activated venv:

```bash
uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000
```

Or without activating the venv:

```bash
uv run --extra train --extra demo uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Start frontend

```bash
cd demo/frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173 — Vite proxies `/api` to the backend.

## Demo workflow

1. Upload `.mp3` or `.wav` (kept as-is; not transcoded)
2. Enter title / artist metadata
3. Click **Estimate BPM / Offset**
4. Adjust **Style** sliders (star rating, pattern ratios, Etterna MSD). Defaults are star rating 0–10, MSD 0–40, analyzer 0–1; the tracks go to 15 / 60 / 1
5. Click **Find matching charts** — picks one real library chart inside all ranges
6. Click **Generate Beatmap** — a progress bar fills while inference runs, then download `.osu` / `.osz`

If no chart matches your ranges, widen the sliders. The demo never invents a condition vector.

## Output layout

Each job is saved under `demo/backend/outputs/<job_id>/`:

```text
<audio sanitized upload name>        # AudioFilename matches this name
Artist - Title (Creator) [Difficulty].osu
Artist - Title.osz
request.json
estimated_timing.json
selected_condition.json
final_cond_vec.json
export_files.json
generation_report.json
error.txt          # on failure
```

`POST /api/generate` returns immediately (`status: running`). Poll `GET /api/jobs/{job_id}` until `completed` or `failed`.

## API endpoints

- `GET /api/health`
- `GET /api/cond-names`
- `POST /api/upload`
- `POST /api/estimate_timing`
- `POST /api/select_condition`
- `POST /api/generate`
- `GET /api/jobs/{job_id}`
- `GET /api/download/{job_id}/osu`
- `GET /api/download/{job_id}/osz`

## Notes

- Model numbers (`cond_dim=18`, 17 user sliders + auto BPM norm): [../README.md](../README.md).
- First inference run computes the audio feature grid and may take several minutes on CPU.
- GPU (`AUDIO2MAP_DEVICE=cuda`) is strongly recommended for reasonable latency.
