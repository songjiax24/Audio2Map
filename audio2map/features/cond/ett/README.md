# Etterna MinaCalc v515

In-process C++ calculator (Etterna 0.74.x, `mina_calc_version = 515`) via
ctypes. Training labels are **SSR at score goal 0.93**, not uncapped MSD mode.

`c_code/` is the pristine tree from
[minacalc-sys 515.2.0](https://crates.io/crates/minacalc-sys) (MIT;
[Glubus/minacalc-rs](https://github.com/Glubus/minacalc-rs)). `bridge.cpp`
exposes `audio2map_minacalc_ssr`.

## Shared library

| Platform | Bundled name |
|----------|----------------|
| Linux | `libminacalc.so` |
| macOS | `libminacalc.dylib` |
| Windows | `minacalc.dll` |

Override with `ETT_MINACALC_LIBRARY`. Rebuild:

```bash
bash audio2map/features/cond/ett/build.sh
```

Windows: same sources, C++20 compiler, `-DSTANDALONE_CALC`, output `minacalc.dll`.

Replace `c_code/` only when bumping the MinaCalc version used for training labels.
Etterna 0.74.1–0.74.4 did not bump `mina_calc_version`.
