# ttsfile

A single-file, cross-platform text-to-speech CLI built on llama.cpp's
[`tools/tts`](https://github.com/ggml-org/llama.cpp/tree/master/tools/tts)
(`llama-tts`) and the same Cosmopolitan packaging as llamafile and
whisperfile. One Actually Portable Executable runs on macOS, Linux, Windows
and BSD, on x86-64 and ARM64, with no installation.

All upstream `llama-tts` flags are accepted, so ttsfile works with the TTS
models the bundled llama.cpp supports. It has been tested with Qwen3-TTS and
Pocket TTS on CPU and GPU; other supported models should work but are
untested here. For per-model options and conversion notes see upstream's
[`tools/tts` README](https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md).

## Quick start

```sh
make setup                                   # once: submodules + patches
.cosmocc/4.0.2/bin/make -j8 o//ttsfile
.cosmocc/4.0.2/bin/make -j8 o//third_party   # needed for self-contained model bundle

HF=https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main
wget $HF/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf
wget $HF/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
wget -O speaker.wav https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav

o//ttsfile/ttsfile \
    -m Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
    -mm mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
    --tts-speaker-file speaker.wav \
    -p "Hello from Qwen3-TTS" -o out.wav
```

Play the result with `afplay out.wav` (macOS) or `aplay out.wav` (Linux).
A speaker reference is recommended for Qwen3-TTS and required for Pocket TTS.
Run with `--help` for the full option list.

## Context Window

The context window defaults to 2048 tokens (`-c` / `--ctx-size` or
`LLAMA_ARG_CTX_SIZE` to change it). At Qwen3-TTS's full 32768-token window the
GPU can run out of memory (`Insufficient Memory`, then `step_gen
failed at frame 1`) because `--fit` does not budget the multimodal
projector. The context bounds one text chunk, not the total audio length. A long
speaker reference uses part of it, so raise `-c` if loading one fails.

## GPU support

Backend loading uses llamafile's runtime loader, as in the other tools: on
Apple Silicon the Metal backend is compiled on first use (needs the Xcode
command-line tools) and cached under `~/.ttsfile/`, separate from
`~/.llamafile/`. Delete that directory to force a rebuild.

To use the CPU, pass `--fit off -ngl 0`. If a GPU run fails with `Insufficient Memory`, use a smaller context
(`-c 2048` or `-c 4096`), add `--no-mmproj-offload` to keep the projector on
the CPU, or fall back to the CPU.

## Self-contained model bundles

Models can be embedded in the executable, llamafile-style, using a `.args`
file for default arguments. This bundles the Quick start files:

```sh
printf -- '-m\n/zip/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf\n-mm\n/zip/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf\n--tts-speaker-file\n/zip/speaker.wav\n...\n' > .args
cp o//ttsfile/ttsfile qwen3-tts-voice.llamafile
o//third_party/zipalign/zipalign -j0 qwen3-tts-voice.llamafile \
    Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
    mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf speaker.wav .args

./qwen3-tts-voice.llamafile -p "Hello" -o out.wav      # no -m needed
```

Command-line arguments still override the embedded defaults. To force the CPU,
add `--fit`, `off`, `-ngl` and `0` as separate lines before the `...` line.

A Pocket TTS bundle works the same way with `pocket-tts-en.gguf` and
`mmproj-pocket-tts-en.gguf` from
[`EryriLabs/pocket-tts-GGUF`](https://huggingface.co/EryriLabs/pocket-tts-GGUF)
and a speaker recording (required); the language is fixed by the weights. The
model files and the speaker recording have separate licenses, so review them
(including attribution requirements) before redistributing a bundle; the
voices' licenses are listed at
[Kyutai's TTS voice collection](https://huggingface.co/kyutai/tts-voices).

## Testing

```sh
tests/ttsfile_smoke.sh o//ttsfile/ttsfile
```

Model-free probes always run; set `TTSFILE_QWEN3_GGUF` and
`TTSFILE_QWEN3_MMPROJ` (and optionally `TTSFILE_SPEAKER`) to enable the
end-to-end layer, which checks the `-p` and `-f` paths and that the output is a
real WAV.

## How it fits together

- `ttsfile/ttsfile.cpp` — entry point: crash reports, `/zip/.args` defaults,
  GPU backend registration and the default context size, then hands argv to
  upstream's TTS CLI (`llama.cpp/tools/tts/tts.cpp`, whose `main` is renamed to
  `tts_cli_main` via `-Dmain=` in `ttsfile/BUILD.mk`).
- `llamafile/gpu_backend.c`, `llamafile/metal.c` — shared GPU probe core and
  Metal runtime build, linked in via `o/$(MODE)/llamafile/gpu.a`.
