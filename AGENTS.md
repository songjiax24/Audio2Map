# Audio2Map agent guide

This file is the standard for new work. It is not a description of v0, and it is
not permission to copy v0 where v0 is messy. v0 stays until a change is in
scope. Do not commit or push unless the user asks.

## Layers

Import direction is one-way, high to low, and `uv run lint-imports` enforces it:

```text
cli → eval → train → generate → dataset → model → features
    → metrics → tokens → grid → osu → utils
```

A layer may import layers below it. It may not import a layer above it, and it
may not reach sideways to skip a documented boundary. `features` does not import
`tokens` or `model`. `osu` does not import `grid`.

One behavior has one implementation. A second copy needs a named owner and a
test that the two outputs match, or it does not land.

## v1 package names

During the transition, versioned code lives in a subpackage, not a prefixed
module:

```text
audio2map/osu/v1/
audio2map/grid/v1/
audio2map/features/audio/v1/
audio2map/tokens/v1/
```

Do not add `timing_v1.py` or `v1_frontend.py` style names. v0 modules stay on
their current paths until a task replaces them. `docs/ARCHITECTURE.md` records
which module is authoritative and what replaces it.

`eval/v1_backend.py` is the v0 evaluation adapter. It is not the v1 tokenizer.

## Constants and errors

A tolerance, sample rate, hop, or reject reason is defined once and imported.
Do not retype `1e-6` or a reason string at the call site.

`ChartRejected` lives in `audio2map/utils/errors.py`. It means the whole chart
cannot be used. Each reason constant maps to one situation. The module is not
an osu parser error list: grid and token structure raise the same type.

`IllegalTokenError` lives in `audio2map/tokens/errors.py`. It means this token
is illegal in the current cursor. It is not a chart-level reject.

Raise the exception from the layer that knows the failure. Do not define it in
a leaf parser and import that parser only to catch the type.

## Names

Modules and functions use plain words (`timing.py`, `label_times`). Types are
`CapWords`. Constants are `UPPER_SNAKE`. Reject reasons are string constants,
not raw literals.

New code stays within 100 columns. No unnamed numeric literal in a formula
when a constant already exists. No function takes 15 or more parameters;
group those values in a small dataclass.

## Logging and imports

Libraries log with `logging.getLogger(__name__)`. They do not `print`.

Lazy imports are only for an optional heavy dependency (`librosa`, `soxr`) at
the function that uses it. Do not hide a layering violation behind a function
import. A trainable module is an `nn.Module` and lives under `audio2map/model/`.
`features/audio/v1` produces decoded audio and normalized mel. It does not own
the stem convolution.

## Tools, scripts, tests

`tools/analyze_*.py` and `tools/compute_*.py` are repo diagnostics: shebang,
argparse, `audio2map` imports. `scripts/` is the data pipeline. Local analysis
that imports `scripts/analysis/` stays untracked.

Tests mirror the package (`tests/osu/v1/`, `tests/grid/v1/`, …). A refactor may
move a test. It may not make an assertion accept more cases or fewer cases
than the behavior it locked.

When a design rule changes, update `docs/v1_design.md` in the same change.
Do not leave the module table pointing at paths that do not exist.

## Git

Do not commit, push, or move a tag unless the user asks. Do not commit
`exports/`, `.local-data/`, or `scripts/analysis/`.
