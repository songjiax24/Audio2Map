# Native v1 generation logits

`generate/v1/trace.py:SamplingStep` is an optional observation of each native
decoder decision. `generate_v1_chart(..., observer=...)` forwards it to the existing
sampler. The callback runs outside CUDA graph capture and receives an independent
float32 CPU copy of the 693 raw logits, before grammar masking or temperature.
It also records the input/chosen token, absolute input-token index, cursor, open
holds, legal ids, and whether the choice is a forced prefix token. Observers must
not consume RNG or mutate model state. The default has no observer.

`generate/v1/sample.py:sampling_distribution` owns the legal-token mask and
temperature transformation for both generation and probability diagnostics. A
zero temperature is a point mass at the same legal argmax. Analysis does not
maintain another implementation of this policy.

`eval/v1/logits.py` owns the recorder, archive loading, raw/conditional ranks and
probabilities. `eval/v1/logit_stats.py` owns reference-onset diagnostics and matched
history/audio comparisons. References use the v1 codec and native exported integer
millisecond clock. Previously covered nearby references are not counted again as
new omissions. These statistics diagnose a known-onset experiment; they do not
assign chart quality or require identical key choices.

## Capture

```bash
uv run python tools/analyze_generation_logits.py \
  --manifest .local-data/processed/eval/generation_logits_20261108/inputs.json \
  --checkpoint .local-data/processed/checkpoints/v1_chart/step_6000.pt \
  --output .local-data/processed/eval/generation_logits_20261108/rerun
```

Use a new output directory when protocol/code changes. Each manifest row names
an audio file/hash, reference chart/hash, `ChartMeta` condition, seed, temperature,
and a unique key. Optional `prefix`/`prefix_sha256` and `history_start` use the
existing replay interface. `silent` zeroes PCM before native encoding. Optional
`expected_tokens`/`expected_tokens_sha256` require exact token reproduction.

Artifacts include frozen source snapshots and hashes, raw `logits.npz`, exact
tokens, exported notes, per-case status, and a completed index. The recorded
2026-11-08 experiment's review script and interpreted artifacts are alongside its
input manifest. Analysis can be rerun from these archives without decoder replay.

## Interpretation

POS selects a position or advances the beat; the following ROW specifies lane
events. A ROW cannot be all empty. Without open holds every legal ROW emits at
least one new head, so an unsupported POS cannot be canceled by an empty ROW.
Previous ROW tokens still condition subsequent POS logits.

For POS decisions, the reference diagnostic asks for the next as-yet-uncovered
reference onset ahead of the current cursor, or BEAT if none remains in that beat.
Equivalent legal POS candidates within 2 ms are pooled. Inspect raw ranks,
legality, conditional probability, and the actual chosen token separately.
Once a prior decision passes an onset, monotonic cursor legality prevents returning
to it. That is distinct from the mask incorrectly rejecting an available target.

For ROW decisions, aggregate probabilities by new-head count and hold starts.
These marginal statistics do not treat the source chart's exact columns as a
uniquely correct arrangement. Source chord sizes are diagnostic context only.

A local legal argmax on an already sampled history is not a full greedy rollout.
The experiment includes separate native greedy generation from BOS. Similarly,
teacher forcing and silent replay keep supplying history tokens; they are not
free generation. Silent replay rebuilds KV under silent audio. Matching token ids
therefore controls the visible symbolic history, not every hidden state.

Human/generated-history comparisons require equal beat, cursor fraction, hold
state, legal-token set and next reference target. They can reveal history
sensitivity but do not isolate rhythm tokens from lane-pattern tokens or prove
which training/model component caused it.
