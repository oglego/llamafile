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
| `response_format` | Optional; only `"wav"` is supported. |
| `model`, `voice` | Accepted for OpenAI-client compatibility and ignored. The voice comes from `--tts-speaker-file`. |

Errors are returned as JSON (`{"error": {"message": ..., "type": ...}}`) with
status 400 for bad requests and 500 for synthesis failures. `GET /health`
returns `{"status":"ok"}`.

## Known gaps

* **Not validated end to end.** Needs a cosmocc build and a run with real
  weights. Real-time factor on CPU is unmeasured; measure it before building
  anything on top of this.
* **No streaming.** A request returns one complete WAV. For conversational use,
  clients should send one sentence per request. The upstream pipeline already
  splits long text into chunks internally and decodes audio per chunk, so a
  streaming API looks feasible, but the public helper currently only exposes the
  accumulated result.
* **GPU use is unverified.** `ttsfile` calls `llamafile_has_gpu()` at startup
  (as `llamafile` itself does) so GPU backends can load, and keeps all inference
  on the main thread, which llamafile's GPU backends require. Whether Metal/CUDA
  actually engages, and whether it is faster for these models, has not been
  measured. Before that call was added the binary ran CPU-only regardless of
  `-ngl`. `ttsfile` uses its own cache directory (`~/.ttsfile/`) for the compiled
  Metal library so it never reuses one built by a regular llamafile.
* **No CORS headers**, so a browser page on another origin cannot call it yet.
* **One voice per server process**, fixed at startup.
* **No request cancellation**: a client that disconnects does not stop its job.
* The synthesis loop in `ttsfile/main.cpp` is a copy of upstream's
  `tools/tts/tts.cpp` loop plus KV-cache/sampler resets between requests. Keep
  the two in step when bumping the llama.cpp submodule.
