# v2 chart tokens (cheat sheet)

> **Unverified agent notes.** Token semantics: **[OVERVIEW.md](OVERVIEW.md)** §6–11.

Full specification: **[OVERVIEW.md](OVERVIEW.md)** (authoritative). Legacy detail may also appear in [v2_spec.md](v2_spec.md) (unverified).

## Sequence shape

```text
<BOS>
<ROW_initial>     # window cut-point hold state; loss=0
<BAR>
<POS_x> <ROW_abcd>
...
<EOS>
```

## Vocab (821)

| Group | Count | Examples |
|-------|-------|----------|
| Special | 4 | `<PAD>`, `<BOS>`, `<EOS>`, `<BAR>` |
| Position | 192 | `<POS_0>` … `<POS_191>` (ticks within one bar) |
| ROW | 625 | `<ROW_0000>` … `<ROW_4444>` (4 lanes × digits 0–4) |

## ROW digit per lane

| Digit | Meaning |
|-------|---------|
| 0 | empty |
| 1 | tap |
| 2 | hold start |
| 3 | hold active |
| 4 | hold end |

Difficulty / SR / pattern info is **not** tokenized — it goes through **cond_vec** (23 floats).

## Code entrypoints

```python
from audio2map.osu.row_tokens import beatmap_to_row_tokens, build_vocab, CanonicalTiming
from audio2map.osu.round_trip import round_trip_beatmap, quantize_notes, tokens_to_notes
from audio2map.training.decode import decode_window_tokens
```

Legacy 10 ms sparse events (removed from training): `audio2map.legacy`.
