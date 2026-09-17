#!/usr/bin/env bash
# Compile MinaCalc v515 into a shared library for ctypes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CXX="${CXX:-g++}"

case "$(uname -s)" in
  Darwin) OUT="$ROOT/libminacalc.dylib" ;;
  *) OUT="$ROOT/libminacalc.so" ;;
esac

"$CXX" -O2 -fPIC -shared -std=c++20 -DSTANDALONE_CALC -w \
  -I "$ROOT/c_code" -I "$ROOT/c_code/MinaCalc" \
  "$ROOT/c_code/API.cpp" \
  "$ROOT/c_code/MinaCalc/MinaCalc.cpp" \
  "$ROOT/bridge.cpp" \
  -o "$OUT"
echo "wrote $OUT"
