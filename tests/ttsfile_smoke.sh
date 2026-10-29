#!/usr/bin/env bash
# ttsfile_smoke.sh — regression smoke for the ttsfile APE.
#
# Two layers:
#   1. Model-free probes — always run. Catches CLI argv / help-text /
#      missing-model regressions without needing any model artifact.
#   2. Qwen3-TTS end-to-end — gated on TTSFILE_QWEN3_GGUF and
#      TTSFILE_QWEN3_MMPROJ. Skipped (with a warning, not a failure) when
#      the models aren't available, so `make check` stays green on machines
#      without them. Exercises the stdin and stdout helpers and asserts the
#      output is a real WAV.
#
# Usage:
#   tests/ttsfile_smoke.sh <path-to-ttsfile-binary>
#
# Env:
#   TTSFILE_QWEN3_GGUF     Path to a Qwen3-TTS backbone GGUF.
#   TTSFILE_QWEN3_MMPROJ   Path to the matching mmproj GGUF.
#   TTSFILE_SPEAKER        Optional speaker reference WAV/MP3.
#   If the GGUF or mmproj is unset or missing, layer 2 is skipped.

set -euo pipefail

APE="${1:-}"
if [ -z "$APE" ] || [ ! -x "$APE" ]; then
    echo "FAIL: argv[1] must point at an executable ttsfile binary" >&2
    echo "      got: '$APE'" >&2
    exit 2
fi

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Every invocation below redirects stdin from /dev/null. ttsfile reads piped
# stdin when no -p is given, so an inherited open pipe (make, CI) would hang.

echo "[smoke] layer 1 — model-free probes"

# --version should exit 0 and print the version banner.
out=$("$APE" --version 2>&1 </dev/null) || fail "--version exited non-zero"
echo "$out" | grep -q '^ttsfile v' || fail "--version missing 'ttsfile v' banner"
pass "--version prints version banner"

# --help should exit 0 and show upstream's usage example.
out=$("$APE" --help 2>&1 </dev/null) || fail "--help exited non-zero"
echo "$out" | grep -q 'text to speak' || fail "--help missing usage example"
pass "--help prints usage"

# A prompt with no model must fail cleanly (non-zero exit, message that
# mentions the model) rather than crash or hang.
if out=$("$APE" -p "hello" -o "$TMP/none.wav" 2>&1 </dev/null); then
    fail "run without -m should exit non-zero"
fi
echo "$out" | grep -qi 'model' || fail "run without -m should mention the missing model"
[ ! -s "$TMP/none.wav" ] || fail "run without -m should not write output"
pass "run without a model fails cleanly"

echo "[smoke] layer 2 — qwen3-tts end-to-end"

MODEL="${TTSFILE_QWEN3_GGUF:-}"
MMPROJ="${TTSFILE_QWEN3_MMPROJ:-}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ] || [ -z "$MMPROJ" ] || [ ! -f "$MMPROJ" ]; then
    echo "  SKIP qwen3-tts end-to-end: TTSFILE_QWEN3_GGUF and" \
         "TTSFILE_QWEN3_MMPROJ not set or files missing" >&2
    echo "       (set both to a Qwen3-TTS GGUF pair to run this layer)" >&2
    exit 0
fi

args=(-m "$MODEL" -mm "$MMPROJ" -n 100)   # ~8 s of audio at 12.5 frames/s
SPEAKER="${TTSFILE_SPEAKER:-}"
if [ -n "$SPEAKER" ] && [ -f "$SPEAKER" ]; then
    args+=(--tts-speaker-file "$SPEAKER")
fi

# Reads the 4-byte little-endian unsigned int at byte offset $2 of $1.
le32() { od -An -tu4 -j"$2" -N4 "$1" | tr -d ' '; }

# Asserts $1 is a RIFF/WAVE file with at least one second of audio, assuming
# 16-bit mono PCM, and prints its duration in seconds.
check_wav() {
    [ -s "$1" ] || fail "$2: output file is empty"
    [ "$(head -c 4 "$1")" = "RIFF" ] || fail "$2: output missing RIFF header"
    [ "$(dd if="$1" bs=1 skip=8 count=4 2>/dev/null)" = "WAVE" ] || fail "$2: output missing WAVE tag"
    rate=$(le32 "$1" 24)
    bytes=$(le32 "$1" 40)
    [ "${rate:-0}" -gt 0 ] || fail "$2: WAV sample rate is zero"
    secs=$((bytes / (rate * 2)))
    [ "$secs" -ge 1 ] || fail "$2: WAV shorter than 1 s (data=$bytes bytes at $rate Hz)"
    echo "$secs"
}

# -p text, file output.
"$APE" "${args[@]}" -p "Hello from the ttsfile smoke test." -o "$TMP/prompt.wav" \
    >/dev/null 2>&1 </dev/null || fail "-p run exited non-zero"
secs=$(check_wav "$TMP/prompt.wav" "-p run")
pass "-p text produces a WAV (${secs}s)"

# Implicit stdin text (no -p) with the WAV written to stdout via -o -.
printf 'Hello from standard input.\n' | "$APE" "${args[@]}" -o - \
    >"$TMP/stdin.wav" 2>/dev/null || fail "stdin run exited non-zero"
secs=$(check_wav "$TMP/stdin.wav" "stdin run")
pass "stdin text to stdout (-o -) produces a WAV (${secs}s)"
