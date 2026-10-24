# ttsfile

Text-to-speech as a single portable executable, built on llama.cpp's
[`tools/tts`](https://github.com/ggml-org/llama.cpp/tree/master/tools/tts)
(`llama-tts`). Supports the models upstream supports (currently Qwen3-TTS and
Pocket TTS).

## Build

```sh
make setup        # applies llama.cpp patches
make -j8 o//ttsfile
```

## Usage

```sh
ttsfile -m model.gguf -mm mmproj.gguf \
    --tts-speaker-file speaker.wav \
    -p "Hello world" -o out.wav
```

All `llama-tts` flags work (`-n`, `--tts-lang`, `--top-k`, `-ngl`, ...).
A speaker reference file is strongly recommended.

## Packaging a single-file voice

Create a `.args` file, one argument per line. The `...` line lets users append
their own arguments (e.g. `-p` and `-o`):

```
-m
/zip/model.gguf
-mm
/zip/mmproj.gguf
--tts-speaker-file
/zip/speaker.wav
...
```

Then embed everything:

```sh
cp o//ttsfile/ttsfile my-voice.llamafile
zipalign -j0 my-voice.llamafile model.gguf mmproj.gguf speaker.wav .args
./my-voice.llamafile -p "Hello world" -o out.wav
```

## How it works

`ttsfile.cpp` mirrors `diffusionfile.cpp`: CPU check, crash reports,
`--version`, load defaults from `/zip/.args`, then call the upstream CLI.
`BUILD.mk` compiles upstream `tts.cpp` with `-Dmain=tts_cli_main` so no patch
to llama.cpp is required.
