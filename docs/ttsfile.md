# ttsfile (experimental)

Text-to-speech built on llama.cpp's multimodal library (libmtmd). It runs the
audio-generation models that upstream llama.cpp supports (Qwen3-TTS, Pocket TTS)
and needs no engine of its own: the model code is already part of the llamafile
build.

> **Status: prototype.** The mtmd audio-generation API is marked experimental
> upstream and may change on llama.cpp syncs. Neither `llama-tts` nor `ttsfile`
> has been built with cosmocc or run against a model as of this writing; see
> "Known gaps".

## Building

Both targets are opt-in and are not part of the default `make`:

```sh
make o//llama.cpp/tts/llama-tts   # upstream's one-shot CLI
make o//ttsfile                   # one-shot CLI + HTTP server
make o//ttsfile/argv_utils_test.runs o//ttsfile/speech_request_test.runs
```

The two unit tests need no model.

## Models and flags

Model files, the matching `--mmproj`, per-model notes, and the `--tts-lang` /
`--tts-speaker-file` flags are documented in llama.cpp's `tools/tts/README.md`.
Some models need a speaker reference clip to produce usable audio; others
support a language code. `ttsfile` accepts the same flags as `llama-tts`.

## One-shot

```sh
ttsfile -m backbone.gguf --mmproj mmproj.gguf -p "Hello there" -o out.wav
```

## Server

```sh
ttsfile -m backbone.gguf --mmproj mmproj.gguf --server --port 8081
```

The server binds to `127.0.0.1` by default and uses port 8081 so it can run
next to llamafile's own server on 8080. It loads the model once. HTTP threads
only queue requests; all synthesis runs one job at a time on the main thread.
Up to 8 requests may wait; beyond that the server answers 503.

```sh
curl -s http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Guten Tag!", "language": "de"}' \
  -o out.wav
```

| Field | Notes |
|---|---|
| `input` | Required. Trimmed; at most 2000 bytes. |
| `language` | Optional two-letter ISO 639-1 code. Falls back to `--tts-lang`. |
| `response_format` | Optional: `"wav"` (default) or `"pcm"` (raw signed 16-bit little-endian mono; the sample rate is in the `X-Sample-Rate` header). |
| `stream` | Optional boolean. `true` sends audio while it is still being generated (see below). |
| `model`, `voice` | Accepted for OpenAI-client compatibility and ignored. The voice comes from `--tts-speaker-file`. |

Errors are returned as JSON (`{"error": {"message": ..., "type": ...}}`) with
status 400 for bad requests and 500 for synthesis failures. `GET /health`
returns `{"status":"ok"}`.

### Speech in, speech out

`POST /v1/audio/speech-to-speech` accepts a `multipart/form-data` upload with
a `file` field containing an audio recording. It transcribes the recording,
sends the transcript to a chat model for a short language-practice reply and
(when appropriate) one gentle correction, then speaks that reply with the
configured TTS model. `--chat-model` and either `--asr-model` or `--asr-url`
are required for this endpoint; ordinary text-to-speech does not need them.

For development, you can run a local Whisper server and provide its base URL.
For the single-file setup, bundle `whisper-server` and its model and pass the
model path with `--asr-model`; ttsfile starts and stops the bundled server:

```sh
ttsfile -m backbone.gguf --mmproj mmproj.gguf --server \
  --asr-model /zip/whisper-model.bin --chat-model /zip/chat-model.gguf
curl -s http://127.0.0.1:8081/v1/audio/speech-to-speech \
  -F 'file=@lesson.wav' -F 'language=de' -F 'session_id=practice-1' -o reply.wav
```

`--asr-model` uses `/zip/whisper-server` by default; use `--asr-executable`
to point to a separately built Whisper server during local development. The
bundled server listens only on `127.0.0.1:8082`. `--asr-url` remains available
to connect to an already-running server.

To assemble one file, build the three targets:

```sh
.cosmocc/4.0.2/bin/make o//ttsfile o//whisperfile/whisper-server o//third_party/zipalign
```

Put this in `.args` (one argument per line):

```text
-m
/zip/tts-backbone.gguf
--mmproj
/zip/tts-mmproj.gguf
--server
--asr-model
/zip/whisper-model.bin
--chat-model
/zip/chat-model.gguf
...
```

Then copy the ttsfile executable and embed its runtime components and weights:

```sh
cp o//ttsfile/ttsfile speech-tutor.llamafile
o//third_party/zipalign/zipalign -0 speech-tutor.llamafile \
  o//whisperfile/whisper-server \
  tts-backbone.gguf tts-mmproj.gguf whisper-model.bin chat-model.gguf
o//third_party/zipalign/zipalign speech-tutor.llamafile .args
```

The Whisper wrapper accepts `--no-default-args` so it does not read the
parent app's `.args` file. The TTS, ASR, and chat models are all loaded at
once, so choose quantizations that fit the machine's memory.

Whisper accepts common audio formats supported by its server, including WAV,
MP3, FLAC, and OGG. The optional `language` form field is a two-letter
ISO 639-1 code. It guides transcription and TTS. When omitted, Whisper detects
the spoken language and ttsfile selects the matching TTS language when the
current Qwen3-TTS model supports it (Chinese, English, German, Italian,
Portuguese, Spanish, Japanese, Korean, French, or Russian); otherwise it uses
the configured `--tts-lang` default. `session_id` is optional; reuse one value
across turns to keep up to three recent exchanges as context. Without it, each
upload is an independent conversation. Transcripts are capped at 1000 bytes
per turn. The audio upload limit is 25 MiB. Uploaded audio is forwarded to the
configured ASR service.

### Streaming

With `"stream": true` the response is sent in chunks as audio is decoded, so
playback can start before the sentence is finished. With `"response_format":
"wav"` the stream starts with a 44-byte WAV header whose length fields are
0xFFFFFFFF ("unknown length"), the usual convention for live streams; players
such as ffplay accept it, but some (and any tool that checks the size) do not.
Use `"pcm"` if you are feeding the audio to your own player.

```sh
curl -sN http://127.0.0.1:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Guten Tag! Wie geht es dir heute?", "language": "de", "stream": true, "response_format": "pcm"}' \
  | ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -
```

How it works: upstream only decodes Qwen3-TTS audio in 72-frame windows (about
5.8 seconds), so a normal sentence would be decoded in one go at the end.
`ttsfile` asks the pipeline to decode every `--stream-chunk-frames` frames
(default 8, about 0.64 s of audio) and sends whatever is new. The decoder keeps
its state between calls, but **whether small windows sound identical to one big
decode has not been verified**. Check it like this, with a fresh server each
time so the random seed starts in the same place:

```sh
# 1. start the server with --seed 42; request without "stream"; save as a.wav
# 2. restart it the same way; request with "stream": true, "response_format": "wav"; save as b.wav
cmp <(tail -c +45 a.wav) <(tail -c +45 b.wav) && echo identical
```

If the two differ, listen for clicks or seams at chunk boundaries and try a
larger `--stream-chunk-frames` (for example 24). Larger values delay the first
audio but reduce the number of decoder calls.

If the client disconnects, the synthesis stops at the next chunk. A failure
before any audio is produced returns a normal 500; a failure after the first
byte aborts the connection, so the client sees a truncated transfer.

## Known gaps

* **Tested on one machine.** A Mac mini (M2, 8 GB) with Qwen3-TTS 1.7B: about
  0.5x real time on CPU and about 0.95x with Metal. Linux, Windows, CUDA, other
  models and Pocket TTS are untested.
* **Streaming is new and lightly tested.** The HTTP framing, progressive
  delivery, disconnect handling and error paths are tested against a real local
  server with a fake audio source. The audio itself, including seams between
  chunks and the real time to first audio, still needs to be checked with a
  model (see "Streaming" above).
* **GPU.** `ttsfile` calls `llamafile_has_gpu()` at startup (as `llamafile`
  does) and keeps all inference on the main thread, which llamafile's GPU
  backends require. Metal is confirmed on Apple silicon only. `ttsfile` uses its
  own cache directory (`~/.ttsfile/`) for the compiled Metal library so it never
  reuses one built by a regular llamafile.
* **No CORS headers**, so a browser page on another origin cannot call it yet.
* **One voice per server process**, fixed at startup.
* **Non-streaming requests cannot be cancelled**: only streaming requests stop
  early when the client disconnects.
* The synthesis loop in `ttsfile/main.cpp` is a copy of upstream's
  `tools/tts/tts.cpp` loop plus KV-cache/sampler resets between requests. Keep
  the two in step when bumping the llama.cpp submodule.
