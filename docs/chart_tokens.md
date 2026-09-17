# Chart tokens

Code: `audio2map/tokens/`. Window framing: `dataset/windows.py`.

## Sequence shape

```text
# decoder window (training / AR generate)
<BOS>
    <ROW_initial>     # hold state at window start; loss=0
<BAR> <POS_x> <ROW_abcd> ...
<EOS>

# chart tokens (encode_notes / tokens_to_notes)
<BAR> <POS_x> <ROW_abcd> ...
```

Decode only the chart body (`unframe_window_tokens`).

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

Difficulty / SR / pattern info is **not** tokenized — it goes through **cond_vec** (18 floats).

## Initial ROW

The token after `<BOS>` is hold state at the window start, not an event. Each lane is only `0` or `3`.

## Empty bars

Every bar emits `<BAR>` even with no events. Empty bars are not `<PAD>`.

## Loss / PAD

| Token | In loss |
|-------|---------|
| `<BOS>`, `<ROW_initial>` | no |
| first `<BAR>` through `<EOS>` | yes |
| `<PAD>` | no (collate only; decode never emits PAD) |

Mask: `dataset/windows.py` (`build_loss_mask`). CE: `train/step.py`.

## Tick identity / decode / export locks

These are current encode/decode/export contracts (locked by `tests/osu/test_osu_export.py` + round-trip tests):

| Step | Behaviour |
|------|-----------|
| `TickNote` | Lattice identity (`grid/tick_note.py`); eval and encode↔decode compare this, not raw ms |
| `tokens_to_notes` | Clip missing head/tail into `notes` + **issues**; every lane (incl. EMPTY / HOLD_ACTIVE) must match hold state or raise |
| `filter_notes_for_export` | Drop `time_ms < 0` and holds with `end <= start` |

Full-chart encode → decode is checked in tests against **TickNote**, not raw ms.

## Code entrypoints

```python
from audio2map.grid import CanonicalTiming, TickNote
from audio2map.tokens import encode_notes, build_vocab, tokens_to_notes
from audio2map.generate.decode import ChartDecodeState
```
