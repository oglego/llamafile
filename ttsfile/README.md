# ttsfile

Single-file, cross-platform text-to-speech CLI built on llama.cpp's `libmtmd`
audio-generation pipelines (Qwen3-TTS, Pocket TTS) and the same Cosmopolitan
packaging as llamafile.

> **Status: proof of concept.** It wraps upstream's `llama.cpp/tools/tts`
> unchanged (that tool is itself a "playground", not production-ready).

## Usage

```sh
# external weights
ttsfile -m pocket-tts.gguf -mm mmproj-pocket-tts.gguf \
    --tts-speaker-file speaker.wav -p "Hello world" -o out.wav

# Qwen3-TTS straight from Hugging Face
ttsfile -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" -o out.wav
```

Context size: unless you pass `-c`, `ttsfile` uses `-c 4096`. Upstream's `tts`
tool would otherwise size the backbone's context to the model's full training
length (32768 for Qwen3-TTS), which reserves about 3.5 GiB of KV cache and was
seen to exhaust GPU memory on an 8 GB Apple Silicon Mac (Metal "Insufficient
Memory" in the audio generation graph). Upstream chose not to change its own
default, since future models may need longer context, so this default lives in
the `ttsfile` wrapper only. Override it with `-c N`, `LLAMA_ARG_CTX_SIZE`, or a
`-c` line in a bundled `.args`. On very small GPUs, `--no-mmproj-offload` keeps
the codec on CPU.

Upstream flags apply (`--tts-lang`, `--tts-speaker-file`, `-n`, `-ngl`, sampling
params). See `llama.cpp/tools/tts/README.md`. `-n` caps the number of frames
(default 512 when unset).

## Bundling weights into one file

Like the other `*file` products, `ttsfile` reads default arguments from
`/zip/.args` inside its own executable:

```sh
cp o//ttsfile/ttsfile my-voice.ttsfile
printf -- '-m\n/zip/model.gguf\n-mm\n/zip/mmproj.gguf\n--tts-speaker-file\n/zip/voice.wav\n' > .args
zipalign -j0 my-voice.ttsfile model.gguf mmproj.gguf voice.wav .args
./my-voice.ttsfile -p "Hello world" -o out.wav
```

## How it is built

- `llama.cpp.patches/llamafile-files/BUILD.mk` compiles `tools/tts/tts.cpp`
  with `-Dmain=tts_cli_main` (no change to the vendored file).
- `ttsfile/main.cpp` owns `main()`: crash reports, `--version`, `/zip/.args`,
  Metal registration, then calls `tts_cli_main()`.
- `libmtmd` was already built for llama-server, so it adds no new sources.
