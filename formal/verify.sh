#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEAN_BIN="${LEAN_BIN:-lean}"
Z3_PYTHON="${Z3_PYTHON:-python3}"

LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT

now_ms() {
    python3 -c 'import time; print(int(time.monotonic() * 1000))'
}

run_check() {
    local label="$1"
    shift
    local t0 t1
    t0=$(now_ms)
    if "$@" >"$LOG_FILE" 2>&1; then
        t1=$(now_ms)
        if grep -qE "uses 'sorry'|warning:" "$LOG_FILE"; then
            printf "[FAIL] %-42s (%d ms) — warning or sorry detected\n" "$label" "$((t1 - t0))"
            cat "$LOG_FILE"
            exit 1
        fi
        printf "[PASS] %-42s (%d ms)\n" "$label" "$((t1 - t0))"
    else
        t1=$(now_ms)
        printf "[FAIL] %-42s (%d ms)\n" "$label" "$((t1 - t0))"
        cat "$LOG_FILE"
        exit 1
    fi
}

for lean_file in \
    "$ROOT/formal/lean/DEOperators.lean" \
    "$ROOT/formal/lean/SearchRunInvariants.lean" \
    "$ROOT/formal/lean/CMAESAndDLR.lean" \
    "$ROOT/formal/lean/NelderMeadAndThermal.lean"
do
    base="$(basename "$lean_file")"
    if grep -qE '^[[:space:]]*axiom[[:space:]]' "$lean_file"; then
        echo "[FAIL] $base declares a custom axiom"
        exit 1
    fi
    run_check "lean $base" "$LEAN_BIN" "$lean_file"
done

read -r -a Z3_CMD <<< "$Z3_PYTHON"
run_check "z3 verify_torch_dfo.py" "${Z3_CMD[@]}" "$ROOT/formal/z3/verify_torch_dfo.py"
echo "All formal verification checks passed."
