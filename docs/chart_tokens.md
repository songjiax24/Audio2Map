# Chart representation (Phase 1 v2)

See the v2 spec in project docs. Summary:

## Token window

```
<BOS>
<ROW_initial>          # prompt only; states {0, 3}
<BAR>                  # empty bars still emit BAR
<POS_x> <ROW_abcd>
...
<EOS>
```

## ROW states (5-state)

| Digit | Meaning |
|-------|---------|
| 0 | empty |
| 1 | tap |
| 2 | hold_start |
| 3 | hold_active |
| 4 | hold_end |

- Vocab: 625 ROW tokens (`<ROW_0000>` … `<ROW_4444>`); model predicts only event rows (≥1 of `{1,2,4}`).
- Initial ROW: 16 combinations of `{0,3}` only.
- Time grid: canonical BPM, 48 ticks/beat, 192 ticks/bar.

## API

```python
from audio2map.osu.row_tokens import beatmap_to_row_tokens, build_vocab, CanonicalTiming
from audio2map.osu.round_trip import round_trip_beatmap, quantize_notes
from audio2map.pattern_analyser import analyze_osu
from audio2map.osu.bpm import canonicalize_bpm
```

Legacy 10ms sparse events remain in `audio2map.osu.events` for archived `processed/`.
