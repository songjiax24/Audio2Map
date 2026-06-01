# Legacy v1 sparse events

**Not used by v2 training or inference.**

This folder holds the old 10 ms frame / sparse event tokenization used in the MVP pipeline (`processed/` on disk). That pipeline was removed from the repo; code here remains only for:

- Parser round-trip tests that reference old token types
- Historical reference when reading [docs/chart_tokens.md](../../docs/chart_tokens.md)

Active v2 code: `audio2map/osu/row_tokens.py`, `audio2map/data/v2_dataset.py`.
