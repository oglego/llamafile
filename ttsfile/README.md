# ttsfile

A single-file, cross-platform text-to-speech CLI built on llama.cpp's
[`tools/tts`](https://github.com/ggml-org/llama.cpp/tree/master/tools/tts)
(`llama-tts`) and the same Cosmopolitan packaging as llamafile and
whisperfile. One Actually Portable Executable runs on macOS, Linux, Windows
and BSD, on x86-64 and ARM64, with no installation.

It supports the models upstream supports (currently Qwen3-TTS and Pocket TTS).

## Quick start

```sh
make setup                                            # once: submodules + patches
.cosmocc/4.0.2/bin/make -j8 o//ttsfile
.cosmocc/4.0.2/bin/make -j8 o//third_party/zipalign   # only needed for bundles

o//ttsfile/ttsfile -m model.gguf -mm mmproj.gguf \
    --tts-speaker-file speaker.wav \
    -p "Hello world" -o out.wav
```

All `llama-tts` flags work (`-n`, `--tts-lang`, `--top-k`, `-ngl`, ...). Run
with `--help` for the full option list.

A speaker reference file (WAV or MP3) is strongly recommended for Qwen3-TTS
and **required** for Pocket TTS, which produces almost no audio without one.

## Output length

Generation stops after 512 frames unless you pass `-n`. `-n` counts audio
frames, not tokens or seconds, and the length of a frame depends on the
model. Qwen3-TTS runs at 12.5 frames per second, so the default cap is about
41 seconds and longer text is cut off silently.

To generate longer audio, set `-n` to 12.5 × the seconds you want:

```sh
ttsfile -m model.gguf -mm mmproj.gguf -p "A long script..." -n 2500 -o out.wav   # ~200 s
```

## Scripting

- `-p -` reads the text to speak from stdin.
- With no `-p` at all, text is read from stdin if it is piped.
- `-o -` writes the WAV to stdout. Logs go to stderr, so stdout carries only
  audio (not supported on Windows).

```sh
cat script.txt | ttsfile -m model.gguf -mm mmproj.gguf -n 2500 -o out.wav
ttsfile -m model.gguf -mm mmproj.gguf -p "Hello" -o - | aplay
```

Because piped stdin is read until EOF, run `ttsfile` with `-p` (or stdin
redirected from `/dev/null`) in contexts where stdin is an open pipe that
never closes, such as some CI runners and process wrappers.

## GPU support

GPU offload uses llamafile's runtime backend loader and llama.cpp's usual
`-ngl` flag. Without a usable GPU, ttsfile runs on CPU.

## Self-contained model bundles

Models can be embedded in the executable, llamafile-style, using a `.args`
file for default arguments. This example uses the Q4_K_M Qwen3-TTS 1.7B
model and its Q8_0 mmproj from
[`ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF).
Run these commands from the repository root, with a speaker-reference WAV
named `speaker.wav`:

```sh
hf download ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    --include Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
    --include mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
    --local-dir qwen3-tts

cat > .args <<'ARGS'
-m
/zip/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf
-mm
/zip/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
--tts-speaker-file
/zip/speaker.wav
-c
2048
-n
2500
...
ARGS

cp o//ttsfile/ttsfile qwen3-tts-voice.llamafile
o//third_party/zipalign/zipalign -j0 qwen3-tts-voice.llamafile \
    qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
    qwen3-tts/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
    speaker.wav .args

./qwen3-tts-voice.llamafile -p "Hello from Qwen3-TTS" -o out.wav   # no -m needed
```

The packaged speaker is the default voice. The trailing `...` line lets
command-line arguments follow the embedded ones, and a later flag overrides
an earlier one, so `-n 500` on the command line wins over the baked-in
`-n 2500`.

## Pocket TTS bundle

This example uses the English Pocket TTS GGUF pair from the community
conversion at [`EryriLabs/pocket-tts-GGUF`](https://huggingface.co/EryriLabs/pocket-tts-GGUF)
and a speaker-reference recording from Kyutai's voice collection. The GGUF
conversion is licensed CC BY 4.0. The `alba-mackenna/casual.wav` recording is
also CC BY 4.0; retain attribution if you redistribute it.

Build `ttsfile` and `zipalign` as in [Quick start](#quick-start), then run
these commands from the repository root:

```sh
hf download EryriLabs/pocket-tts-GGUF \
    --include pocket-tts-en.gguf \
    --include mmproj-pocket-tts-en.gguf \
    --local-dir pocket-tts

curl -L \
    https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav \
    -o speaker.wav

cat > .args <<'ARGS'
-m
/zip/pocket-tts-en.gguf
-mm
/zip/mmproj-pocket-tts-en.gguf
--tts-speaker-file
/zip/speaker.wav
...
ARGS

cp o//ttsfile/ttsfile pocket-tts-voice.llamafile
o//third_party/zipalign/zipalign -j0 pocket-tts-voice.llamafile \
    pocket-tts/pocket-tts-en.gguf \
    pocket-tts/mmproj-pocket-tts-en.gguf \
    speaker.wav .args

./pocket-tts-voice.llamafile -p "Hello from Pocket TTS" -o out.wav
```

The sample voice is hosted in [Kyutai's TTS voice collection](https://huggingface.co/kyutai/tts-voices), which lists the licenses for its recordings. Pocket TTS requires a speaker-reference file; the English model uses English text.

The default 512-frame cap applies here too (see [Output length](#output-length)).
Add `-n` to `.args` or the command line for longer text. How many seconds a
frame covers depends on the model, so check the length of your output.

## Package without a speaker reference

Qwen3-TTS Base can also synthesize without a speaker-reference file, but using
one gives better results. For a smaller, simpler bundle, create `.args` without
`--tts-speaker-file`:

```sh
cat > .args <<'EOF'
-m
/zip/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf
-mm
/zip/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
-c
2048
...
EOF

cp o//ttsfile/ttsfile qwen3-tts-no-reference.llamafile
o//third_party/zipalign/zipalign -j0 qwen3-tts-no-reference.llamafile \
    qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
    qwen3-tts/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
    .args
```

Run the bundled executable with text and an output path:

```sh
./qwen3-tts-no-reference.llamafile -p "Hello from Qwen3-TTS" -o out.wav
```

## How it fits together

- `ttsfile/ttsfile.cpp` is the entry point: CPU check, crash reports,
  `--version`, `/zip/.args` defaults, GPU backend registration, the stdin and
  stdout helpers above, then it hands argv to upstream's CLI.
- `ttsfile/BUILD.mk` compiles upstream `llama.cpp/tools/tts/tts.cpp` with
  `-Dmain=tts_cli_main` so its `main` can be called from `ttsfile.cpp`. No
  patch to llama.cpp is required.
- Linking needs `$(HTTPLIB_OBJS)` and `mbedtls.a` alongside `llama.cpp.a`
  because upstream `common` is built with `-DLLAMA_USE_HTTPLIB` (for `-hf`
  downloads).
