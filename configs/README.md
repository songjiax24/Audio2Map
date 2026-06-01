# Config files (reference only)

Scripts use **argparse** today; these YAML files are **not loaded automatically**.  
They document intended hyperparameters for copy-paste into CLI flags.

| File | Maps to |
|------|---------|
| `data/v2.yaml` | Window bars, margins, audio sample rate |
| `data/collector.yaml` | Sayobot download delays |
| `train/train_multi.yaml` | **Formal** `train_v2.py` multi-chart training |
| `train/debug_overfit.yaml` | `train_debug_overfit.py` (debug only) |

## Formal training (current)

Precompute is **complete** (11,927/11,927 charts have valid grid). Defaults in `train_v2.py`:

```bash
python scripts/train_v2.py
# → checkpoints/train_v1/, audio_pooling=tick, steps=50000, require_grid=true
```

Legacy trial checkpoints live under `checkpoints/trial_multi_v2/` (bar pooling, partial-data era).
