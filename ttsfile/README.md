# ttsfile

Text-to-speech as a single portable executable, built on llama.cpp's
[`tools/tts`](https://github.com/ggml-org/llama.cpp/tree/master/tools/tts)
(`llama-tts`). Supports the models upstream supports (currently Qwen3-TTS and
Pocket TTS).

## Build

```sh
make setup        # applies llama.cpp patches
.cosmocc/4.0.2/bin/make -j8
```

## Usage

```sh
ttsfile -m model.gguf -mm mmproj.gguf \
    --tts-speaker-file speaker.wav \
    -p "Hello world" -o out.wav
```

All `llama-tts` flags work (`-n`, `--tts-lang`, `--top-k`, `-ngl`, ...).
A speaker reference file is strongly recommended.

## Package a Qwen3-TTS single-voice executable

This example uses the Q4_K_M Qwen3-TTS 1.7B model and its Q8_0 mmproj from
[`ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF).
Run these commands from the repository root. Download the model files and
provide a speaker-reference WAV named `speaker.wav`:

```sh
make setup
.cosmocc/4.0.2/bin/make -j8
hf download ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    --include Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
    --include mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
    --local-dir qwen3-tts
```

Create `.args` with one argument per line. The `...` line allows users to
append runtime options:

```sh
cat > .args <<'EOF'
-m
/zip/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf
-mm
/zip/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
--tts-speaker-file
/zip/speaker.wav
-c
2048
...
EOF
```

Embed the executable, models, speaker WAV, and defaults:

```sh
cp o//ttsfile/ttsfile qwen3-tts-voice.llamafile
o//third_party/zipalign/zipalign -j0 qwen3-tts-voice.llamafile \
    qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
    qwen3-tts/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
    speaker.wav .args
```

The embedded arguments work for both CLI and server mode. Run the packaged
voice as a server and request speech:

```sh
./qwen3-tts-voice.llamafile --server --port 8080
```

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Qwen3-TTS","response_format":"wav"}' \
  -o out.wav
```

For one-shot CLI use, append prompt and output options instead:

```sh
./qwen3-tts-voice.llamafile -p "Hello from Qwen3-TTS" -o out.wav
```

The packaged speaker is the default voice. To select among multiple voices at
request time, use an external `--voices-dir` instead of bundling just one
speaker.

### Package without a speaker reference

Qwen3-TTS Base can also synthesize without a speaker-reference file, although
providing one is recommended for better results. For a smaller, simpler bundle,
create `.args` without `--tts-speaker-file`:

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

Run it the same way as the speaker-based bundle, for example:

```sh
./qwen3-tts-no-reference.llamafile --server --port 8080
```

## How it works

`ttsfile.cpp` mirrors `diffusionfile.cpp`: CPU check, crash reports,
`--version`, load defaults from `/zip/.args`, then call the upstream CLI.
`BUILD.mk` compiles upstream `tts.cpp` with `-Dmain=tts_cli_main` so no patch
to llama.cpp is required.

## Scripting: stdin and stdout

```sh
echo "Hello world" | ttsfile -m model.gguf -mm mmproj.gguf -o out.wav   # text from stdin
cat story.txt     | ttsfile ... -p - -o out.wav                         # explicit form
echo "Hello"      | ttsfile ... -o - | ffplay -autoexit -nodisp -      # WAV to stdout
```

If `-p` is not given and stdin is piped, the text is read from stdin. `-o -`
writes the WAV to stdout (logs always go to stderr). Not tested on Windows.

## Server mode

`--server` keeps the model loaded and serves an OpenAI-compatible endpoint:

```sh
ttsfile --server -m model.gguf -mm mmproj.gguf \
    --tts-speaker-file speaker.wav -c 4096 --port 8080
```

For a packaged voice, start the executable with `--server` and any desired
server flags; its model, mmproj, and default speaker are loaded from the
embedded `.args` defaults.

```sh
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "response_format": "wav"}' -o out.wav
```

Request fields:

| Field | Notes |
| --- | --- |
| `input` | Required. Text to speak (default limit 4096 chars, `--max-input-chars`). |
| `voice` | Name of `<voices-dir>/<voice>.wav`. Without `--voices-dir`, ignored and the `--tts-speaker-file` voice is used. |
| `response_format` | `wav` (default) or `pcm` (16-bit LE mono). mp3/opus/aac/flac are not supported. |
| `stream` | `true` streams audio as it is generated (chunked). Only raw audio streaming; `stream_format: "sse"` is rejected. |
| `language` | Extension: overrides `--tts-lang` for this request. |
| `model`, `speed`, `instructions` | Accepted and ignored. |

Server flags: `--host` (default 127.0.0.1), `--port` (8080), `--voices-dir`,
`--api-key` (require `Authorization: Bearer <key>`), `--max-input-chars`.
Also `GET /health` and `GET /v1/models`. Requests are generated one at a time.

Streaming playback example (PCM at the model's sample rate, 24 kHz for the
models tested upstream; check yours):

```sh
curl -sN http://127.0.0.1:8080/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input":"Hello","response_format":"pcm","stream":true}' \
  | ffplay -f s16le -ar 24000 -nodisp -autoexit -
```

Playback is left to the player rather than built in, so the binary stays
portable across platforms.
