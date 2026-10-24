#!/usr/bin/env bash
# ttsfile_smoke.sh — regression smoke for the ttsfile APE.
#
# Two layers:
#   1. Model-free probes — always run. Catches CLI argv / help-text /
#      missing-model regressions without needing any model artifact.
#   2. Qwen3-TTS end-to-end — gated on TTSFILE_QWEN3_GGUF and
#      TTSFILE_QWEN3_MMPROJ. Skipped (with a warning, not a failure) when
#      the models aren't available, so `make check` stays green on machines
#      without them. Asserts the output is a real WAV, and covers the -p and
#      -f paths plus the default context size.
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

echo "[smoke] layer 1 — model-free probes"

# --help should exit 0 and show upstream's usage example.
out=$("$APE" --help 2>&1 </dev/null) || fail "--help exited non-zero"
grep -q 'text to speak' <<<"$out" || fail "--help missing usage example"
pass "--help prints usage"

# A prompt with no model must fail cleanly (non-zero exit, message that
# mentions the model) rather than crash or hang.
if out=$("$APE" -p "hello" -o "$TMP/none.wav" 2>&1 </dev/null); then
    fail "run without -m should exit non-zero"
fi
grep -qi 'model' <<<"$out" || fail "run without -m should mention the missing model"
[ ! -s "$TMP/none.wav" ] || fail "run without -m should not write output"
pass "run without a model fails cleanly"

if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    if command -v xcode-select >/dev/null 2>&1 && xcode-select -p >/dev/null 2>&1; then
        out=$("$APE" --list-devices 2>&1 </dev/null) || fail "--list-devices exited non-zero"
        grep -qi 'metal' <<<"$out" || fail "--list-devices did not register the Metal backend"
        pass "--list-devices registers the Metal backend"
    else
        echo "  SKIP Metal device listing: Xcode command-line tools unavailable" >&2
    fi
fi

echo "[smoke] layer 2 — qwen3-tts end-to-end"

MODEL="${TTSFILE_QWEN3_GGUF:-}"
MMPROJ="${TTSFILE_QWEN3_MMPROJ:-}"
missing=()
for v in TTSFILE_QWEN3_GGUF TTSFILE_QWEN3_MMPROJ; do
    p="${!v:-}"
    if [ -z "$p" ]; then
        missing+=("$v is not set")
    elif [ ! -f "$p" ]; then
        missing+=("$v: no such file '$p'")
    fi
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "  SKIP qwen3-tts end-to-end:" >&2
    printf '       %s\n' "${missing[@]}" >&2
    echo "       (set both to a Qwen3-TTS GGUF pair to run this layer)" >&2
    exit 0
fi

# -n 100 is ~8 s of audio at 12.5 frames/s. Leave -c unset so this smoke test
# exercises ttsfile's default 2048-token context window, which avoids the GPU
# out-of-memory failure seen when llama.cpp uses the model's full context.
args=(-m "$MODEL" -mm "$MMPROJ" -n 100)
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
if ! "$APE" "${args[@]}" -p "Hello from the ttsfile smoke test." -o "$TMP/prompt.wav" \
        >"$TMP/prompt.log" 2>&1 </dev/null; then
    tail -n 20 "$TMP/prompt.log" >&2
    fail "-p run exited non-zero"
fi
secs=$(check_wav "$TMP/prompt.wav" "-p run")
pass "-p text produces a WAV (${secs}s)"

# -f supplies the prompt from a file.
printf 'Hello from a prompt file.\n' >"$TMP/prompt.txt"
if ! "$APE" "${args[@]}" -f "$TMP/prompt.txt" -o "$TMP/file.wav" \
        >"$TMP/file.log" 2>&1 </dev/null; then
    tail -n 20 "$TMP/file.log" >&2
    fail "-f run exited non-zero"
fi
secs=$(check_wav "$TMP/file.wav" "-f run")
pass "-f prompt file produces a WAV (${secs}s)"
