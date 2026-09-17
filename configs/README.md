# Config files (loaded by `--config`)

Every product CLI entry point (`audio2map-meta / -grid / -train / -eval / -infer`)
accepts `--config path/to.yaml`. The YAML mapping supplies **defaults** for
the matching argparse options; explicit CLI flags always win. Keys that do not match an
option of that command are ignored with a warning on stderr, except keys passed as
``locked`` (must match the code constant if present).

`python -m scripts.collect.cli` also accepts `--config` (data-prep script, not a
product entry point).

| File | Maps to |
|------|---------|
| `data/v2.yaml` | `audio2map-infer` overlap constants (`window_bars` 16 = 8+4+4, `max_seq_len` 2048). Locked; cannot disagree with `OverlapConfig`. |
| `data/collector.yaml` | `python -m scripts.collect.cli` (`page_size`, `list_delay`) |
| `train/train_multi.yaml` | `audio2map-train` formal multi-chart training |
| `train/debug_overfit.yaml` | Reference values for `scripts/train_debug_overfit.py` (that script does not load YAML) |

## Formal training (current)

```bash
audio2map-train --config configs/train/train_multi.yaml
# → processed/checkpoints/formal_enc_dec_v3/
#    max_steps=200000, batch_size=16, d_model=512,
#    encoder_layers=4, decoder_layers=6, heads=8,
#    samples_per_chart=4, window=16 bars (tick-level audio)
```

`n_heads` in YAML is accepted as an alias for `heads`. Override anything ad hoc, e.g.
`audio2map-train --config configs/train/train_multi.yaml --max-steps 500 --limit 8`.
`window_bars` must equal `WINDOW_BARS` (16) if present; it is not a training flag.

Legacy trial dirs such as `checkpoints/trial_multi_v2/` or old `train_v1/` are historical only.
