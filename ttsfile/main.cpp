// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// ttsfile - text-to-speech on top of llama.cpp's libmtmd audio generation
// (Qwen3-TTS, Pocket TTS).
//
//   ttsfile -m model.gguf --mmproj mmproj.gguf -p "Hello" -o out.wav
//   ttsfile -m model.gguf --mmproj mmproj.gguf --server --port 8081
//
// One-shot mode mirrors upstream's llama-tts. Server mode loads the model
// once and serves POST /v1/audio/speech (WAV out) and GET /health.
//
// EXPERIMENTAL. The mtmd audio-generation API this builds on is marked
// experimental upstream and may break on llama.cpp syncs.
//
// The synthesis loop is deliberately a close copy of llama.cpp/tools/tts/
// tts.cpp so that behaviour matches llama-tts; keep the two in step when
// bumping the llama.cpp submodule.
//

#include "argv_utils.h"
#include "audio_channel.h"
#include "audio_encode.h"
#include "job_queue.h"
#include "stream_response.h"
#include "speech_request.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <spawn.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#ifdef COSMOCC
#include <cosmo.h>
#include "llamafile/llamafile.h"
#endif

#ifndef TTSFILE_VERSION_STRING
#define TTSFILE_VERSION_STRING "0.0.0-dev"
#endif

extern char ** environ;

namespace {

void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m backbone.gguf --mmproj mmproj.gguf -p \"text to speak\" -o output.wav", argv[0]);
    LOG("\n    %s -hf user/model -p \"text to speak\" -o output.wav", argv[0]);
    LOG("\n    %s -hf user/model --server --port 8081\n", argv[0]);
    LOG("\nttsfile options:\n");
    LOG("    --server          serve POST /v1/audio/speech instead of one-shot synthesis\n");
    LOG("    --host ADDR       server bind address (default: 127.0.0.1)\n");
    LOG("    --port PORT       server port (default: 8081; llamafile itself uses 8080)\n");
    LOG("    --stream-chunk-frames N\n");
    LOG("                      server streaming: decode and send audio every N frames\n");
    LOG("                      (default: 8). Smaller = lower first-audio delay but more\n");
    LOG("                      decoder calls; must be tested for audible seams.\n");
    LOG("    --asr-url URL     Whisper server base URL for speech-to-speech\n");
    LOG("    --asr-model GGUF  start the bundled whisper-server with this ASR model\n");
    LOG("    --asr-executable PATH\n");
    LOG("                      whisper-server path (default: /zip/whisper-server)\n");
    LOG("    --chat-model GGUF conversational model for the speech tutor\n");
    LOG("    --version         print version and exit\n");
    LOG("\nnotes: --tts-lang and --tts-speaker-file may not be supported by all models;\n");
    LOG("       see llama.cpp's tools/tts/README.md for per-model usage notes.\n");
    LOG("       default arguments are read from /zip/.args when bundled with zipalign.\n\n");
}

// ---------------------------------------------------------------------------
// synthesis engine: everything loaded once, one request at a time
// ---------------------------------------------------------------------------

struct tts_engine {
    common_params      * params = nullptr;
    llama_context      * lctx   = nullptr;
    mtmd_context       * mctx   = nullptr;
    common_sampler     * smpl   = nullptr;
    mtmd_bitmap        * speaker = nullptr;  // optional reference voice, owned by caller
    int                  sample_rate = 0;    // Hz, from the model
    int                  stream_chunk_frames = 8;
};

struct tutor_turn {
    std::string user;
    std::string assistant;
};

// The chat model is optional for text-to-speech, but required for the tutor
// endpoint. It is separate from the speech-generation model: Qwen3-TTS is a
// speech synthesizer, not a general conversational assistant.
struct tutor_engine {
    common_init_result_ptr    init;
    std::unordered_map<std::string, std::deque<tutor_turn>> sessions;
};

std::string tutor_language_name(const std::string & code) {
    static const std::pair<const char *, const char *> names[] = {
        {"zh", "Chinese"}, {"en", "English"}, {"de", "German"}, {"it", "Italian"},
        {"pt", "Portuguese"}, {"es", "Spanish"}, {"ja", "Japanese"}, {"ko", "Korean"},
        {"fr", "French"}, {"ru", "Russian"},
    };
    for (const auto & item : names) if (code == item.first) return item.second;
    return code;
}

bool generate_tutor_reply(tutor_engine & e, const std::string & text, const std::string & lang,
                          const std::string & session_id, std::string & reply, std::string * err) {
    if (!e.init || !e.init->model() || !e.init->context() || !e.init->sampler(0)) {
        *err = "tutor chat model is not initialized";
        return false;
    }
    llama_model * model = e.init->model();
    llama_context * ctx = e.init->context();
    common_sampler * smpl = e.init->sampler(0);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string language = lang.empty() ? "the language used by the learner" : tutor_language_name(lang);
    std::string system = "You are a friendly language-practice partner. Reply in " + language +
        ". Respond naturally to what the learner said, then include at most one brief, gentle correction "
        "if their wording has a clear mistake. Keep the entire spoken reply to one or two short sentences. "
        "Use plain text only: no markdown, lists, labels, or explanations about being an AI.";

    std::deque<tutor_turn> & history = e.sessions[session_id];
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    roles.reserve(2 + history.size() * 2);
    contents.reserve(2 + history.size() * 2);
    roles.push_back("system");
    contents.push_back(std::move(system));
    for (const tutor_turn & turn : history) {
        roles.push_back("user");
        contents.push_back(turn.user);
        roles.push_back("assistant");
        contents.push_back(turn.assistant);
    }
    roles.push_back("user");
    contents.push_back(text);

    std::vector<llama_chat_message> messages;
    messages.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    }
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        *err = "tutor model has no chat template";
        return false;
    }
    std::vector<char> formatted(32768);
    int32_t formatted_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                       formatted.data(), formatted.size());
    if (formatted_len < 0) {
        *err = "could not format tutor chat prompt";
        return false;
    }
    if ((size_t) formatted_len > formatted.size()) {
        formatted.resize((size_t) formatted_len + 1);
        formatted_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                  formatted.data(), formatted.size());
        if (formatted_len < 0) {
            *err = "could not format tutor chat prompt";
            return false;
        }
    }
    std::string prompt(formatted.data(), (size_t) formatted_len);
    const int32_t n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, false, true);
    if (n_prompt <= 0 || n_prompt > llama_n_ctx(ctx)) {
        *err = "tutor prompt is empty or exceeds the chat model context";
        return false;
    }
    std::vector<llama_token> tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), false, true) < 0) {
        *err = "could not tokenize tutor prompt";
        return false;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    common_sampler_reset(smpl);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token token = LLAMA_TOKEN_NULL;
    reply.clear();
    for (int i = 0; i < 160; ++i) {
        if (llama_decode(ctx, batch) != 0) {
            *err = "tutor chat model decode failed";
            return false;
        }
        token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, token, true);
        if (llama_vocab_is_eog(vocab, token)) break;
        char piece[4096];
        const int n_piece = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (n_piece < 0) {
            *err = "could not decode tutor response token";
            return false;
        }
        reply.append(piece, (size_t) n_piece);
        batch = llama_batch_get_one(&token, 1);
    }
    reply = ttsfile::detail::trim(reply);
    if (reply.empty()) {
        *err = "tutor chat model returned an empty reply";
        return false;
    }
    if (!session_id.empty()) {
        history.push_back({text, reply});
        while (history.size() > 3) history.pop_front();
        if (e.sessions.size() > 128) e.sessions.erase(e.sessions.begin());
    }
    return true;
}

struct tts_result {
    std::string wav;
    int         n_frames    = 0;
    int32_t     sample_rate = 0;
    int64_t     n_samples   = 0;
    double      t_total_s   = 0.0;
    double      t_first_audio_s = -1.0;  // streaming only; -1 = no audio yet
    bool        cancelled   = false;     // streaming: the consumer went away
};

// Streaming output. emit() receives new float samples as they are decoded and
// returns false if the consumer is gone, which stops the synthesis.
struct stream_sink {
    int                                              chunk_frames = 8;
    std::function<bool(const float *, size_t)>       emit;
};

// Synthesizes `text` and fills `res`. Returns false (with a message in *err)
// on failure. Not thread-safe: callers must serialize access to the engine.
bool synthesize(tts_engine & e, const std::string & text, const std::string & lang,
                tts_result & res, std::string * err, const stream_sink * sink = nullptr) {
    const common_params & params = *e.params;
    const int64_t t_start_us = ggml_time_us();

    // Each request starts from a clean slate. This is required, not just
    // tidy: the pipeline's reset()/set_input() rewinds its own position
    // counter to 0 but does not touch llama's KV cache, and the one-shot
    // llama-tts never notices because it exits after a single request. A
    // server that reused the context as-is would decode the next prompt on
    // top of the previous request's stale KV entries. The sampler history is
    // cleared for the same reason.
    llama_memory_clear(llama_get_memory(e.lctx), true);
    common_sampler_reset(e.smpl);

    mtmd_helper::gen_audio gen(e.lctx, e.mctx);
    if (!gen.ctx) {
        *err = "failed to create audio generation context";
        return false;
    }

    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id      = 0;
    inp.prompt      = text.c_str();
    inp.prompt_len  = text.size();
    inp.speaker_ref = e.speaker;
    inp.lang        = lang.c_str();
    inp.top_k       = params.sampling.top_k;
    inp.top_p       = params.sampling.top_p;
    inp.seed        = params.sampling.seed;
    inp.out_type    = sink ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM : MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;

    // stage 1: process the prompt through the backbone
    if (gen.set_input(&inp) != 0) {
        *err = "set_input failed";
        return false;
    }
    for (;;) {
        const int32_t ret = gen.step_prompt(params.n_batch);
        if (ret < 0) {
            *err = "prompt processing failed";
            return false;
        }
        if (ret == 0) {
            break;
        }
    }

    auto sample_semantic_code = [&]() -> llama_token {
        llama_token t = common_sampler_sample(e.smpl, e.lctx, -1);
        common_sampler_accept(e.smpl, t, true);
        return t;
    };

    // Streaming support. The pipeline appends decoded audio to one cumulative
    // buffer, and get_output() both decodes any frames still pending and
    // returns that whole buffer (in PCM mode, the float samples in place, no
    // copy). So "new audio" is everything past what was already emitted.
    //
    // Upstream decodes Qwen3-TTS audio in 72-frame windows (about 5.8s), so a
    // normal sentence would only be decoded at the very end. Calling
    // get_output() every few frames forces earlier decodes. The decoder state
    // is carried between calls, but whether small windows sound identical to
    // one big decode is untested: see the A/B check in docs/ttsfile.md.
    int64_t emitted = 0;
    enum emit_rc { EMIT_OK, EMIT_ERROR, EMIT_CANCELLED };
    auto emit_new = [&]() -> emit_rc {
        int32_t      sr  = 0;
        const char * d   = nullptr;
        size_t       len = 0;
        int64_t      n   = 0;
        if (gen.get_output(&sr, &d, &len, &n) != 0) {
            *err = "audio decode failed";
            return EMIT_ERROR;
        }
        if (n > emitted) {
            if (res.t_first_audio_s < 0) {
                res.t_first_audio_s = (ggml_time_us() - t_start_us) / 1e6;
            }
            if (!sink->emit(reinterpret_cast<const float *>(d) + emitted, (size_t) (n - emitted))) {
                return EMIT_CANCELLED;
            }
            emitted = n;
        }
        return EMIT_OK;
    };

    // stage 2+3: semantic code -> acoustic detail -> waveform, frame by frame
    const int max_new = params.n_predict > 0 ? params.n_predict : 512;
    int n_frames = 0;
    llama_token   sampled = sample_semantic_code();
    const float * h_state = llama_get_embeddings_ith(e.lctx, -1);

    bool stop = false;
    while (!stop && n_frames < max_new) {
        const float * h_next = nullptr;
        if (gen.step_gen(sampled, h_state, &h_next, &stop) != 0) {
            *err = "step_gen failed at frame " + std::to_string(n_frames);
            return false;
        }
        if (!h_next) {
            break;  // stopped without generating a frame
        }
        ++n_frames;
        h_state = h_next;
        if (sink && n_frames % sink->chunk_frames == 0) {
            const emit_rc rc = emit_new();
            if (rc == EMIT_CANCELLED) {
                res.cancelled = true;
                *err = "client disconnected";
                return false;
            }
            if (rc == EMIT_ERROR) {
                return false;
            }
        }
        sampled = sample_semantic_code();
    }

    if (sink) {
        const emit_rc rc = emit_new();  // whatever is left after the last full chunk
        if (rc == EMIT_CANCELLED) {
            res.cancelled = true;
            *err = "client disconnected";
            return false;
        }
        if (rc == EMIT_ERROR) {
            return false;
        }
        if (emitted == 0) {
            *err = "model produced no audio";
            return false;
        }
        res.n_frames    = n_frames;
        res.sample_rate = e.sample_rate;
        res.n_samples   = emitted;
        res.t_total_s   = (ggml_time_us() - t_start_us) / 1e6;
        const double audio_s = e.sample_rate > 0 ? (double) emitted / e.sample_rate : 0.0;
        LOG_INF("streamed %d frames -> %.2fs of audio in %.2fs (audio time = %.2fx process time), first audio after %.2fs\n",
                n_frames, audio_s, res.t_total_s, res.t_total_s > 0 ? audio_s / res.t_total_s : 0.0,
                res.t_first_audio_s);
        return true;
    }

    int32_t      sample_rate = 0;
    const char * data        = nullptr;
    size_t       data_len    = 0;
    int64_t      n_samples   = 0;
    if (gen.get_output(&sample_rate, &data, &data_len, &n_samples) != 0 || !data || data_len == 0) {
        *err = "model produced no audio";
        return false;
    }

    res.wav.assign(data, data_len);  // `data` dies with the next get_output()/reset()
    res.n_frames    = n_frames;
    res.sample_rate = sample_rate;
    res.n_samples   = n_samples;
    res.t_total_s   = (ggml_time_us() - t_start_us) / 1e6;

    const double audio_s = sample_rate > 0 ? (double) n_samples / sample_rate : 0.0;
    LOG_INF("synthesized %d frames -> %.2fs of audio in %.2fs (audio time = %.2fx process time)\n",
            n_frames, audio_s, res.t_total_s, res.t_total_s > 0 ? audio_s / res.t_total_s : 0.0);
    return true;
}

// ---------------------------------------------------------------------------
// HTTP server
//
// Threading model (same idea as llama-server): httplib worker threads only
// parse requests and enqueue jobs. ALL synthesis runs on the thread that
// loaded the model (main), one job at a time. llamafile's GPU backends must
// run on the main thread (see the yield_to_queue() comment in the server-queue
// patch), and an early version of this server that ran the model on httplib
// worker threads died with a bare "bus error". The root cause of that crash
// was never confirmed, so this is also the conservative design.
// ---------------------------------------------------------------------------

constexpr size_t kMaxQueuedJobs = 8;  // beyond this, requests get 503 instead of piling up

struct job_outcome {
    bool        ok = false;
    std::string err;
    tts_result  result;
};

struct tts_job {
    std::string                text;
    std::string                lang;
    std::string                session_id;
    bool                       tutor = false;
    // exactly one of these is used:
    std::promise<job_outcome>                done;     // non-streaming: whole result
    std::shared_ptr<ttsfile::audio_channel>  channel;  // streaming: chunks as produced
};

// Runs one job on the main thread.
void run_job(tts_engine & e, tutor_engine * tutor, tts_job & job) {
    if (!job.channel) {
        job_outcome out;
        if (job.tutor) {
            std::string response;
            if (!tutor || !generate_tutor_reply(*tutor, job.text, job.lang, job.session_id, response, &out.err)) {
                out.ok = false;
                if (out.err.empty()) out.err = "speech tutor requires --chat-model";
                job.done.set_value(std::move(out));
                return;
            }
            job.text = std::move(response);
        }
        out.ok = synthesize(e, job.text, job.lang, out.result, &out.err);
        job.done.set_value(std::move(out));
        return;
    }

    ttsfile::audio_channel & ch = *job.channel;
    if (ch.cancelled()) {  // the client left while this was still queued
        ch.finish();
        return;
    }

    stream_sink sink;
    sink.chunk_frames = e.stream_chunk_frames;
    sink.emit = [&ch](const float * pcm, size_t n) {
        std::string bytes;
        ttsfile::append_s16le(bytes, pcm, n);
        return ch.push(std::move(bytes));
    };

    tts_result  result;
    std::string err;
    if (synthesize(e, job.text, job.lang, result, &err, &sink)) {
        ch.finish();
    } else if (result.cancelled) {
        LOG_INF("client disconnected; synthesis stopped early\n");
        ch.finish();
    } else {
        LOG_ERR("streaming synthesis failed: %s\n", err.c_str());
        ch.fail(err);
    }
}

std::atomic<httplib::Server *> g_server{nullptr};

std::string tts_language_from_asr(const nlohmann::json & transcript) {
    if (!transcript.contains("language") || !transcript["language"].is_string()) return "";
    std::string name = transcript["language"].get<std::string>();
    for (char & c : name) c = (char) std::tolower((unsigned char) c);
    // Qwen3-TTS currently has language tokens for these ten languages.
    // Whisper returns their full names in verbose_json responses.
    static const std::pair<const char *, const char *> supported[] = {
        {"chinese", "zh"}, {"english", "en"}, {"german", "de"},
        {"italian", "it"}, {"portuguese", "pt"}, {"spanish", "es"},
        {"japanese", "ja"}, {"korean", "ko"}, {"french", "fr"}, {"russian", "ru"},
    };
    for (const auto & entry : supported) {
        if (name == entry.first) return entry.second;
    }
    return "";
}

constexpr int kEmbeddedAsrPort = 8082;

bool start_embedded_asr(const std::string & executable, const std::string & model_path,
                        pid_t & pid, std::string & err) {
    char * argv[] = {
        const_cast<char *>(executable.c_str()),
        const_cast<char *>("-m"), const_cast<char *>(model_path.c_str()),
        const_cast<char *>("--host"), const_cast<char *>("127.0.0.1"),
        const_cast<char *>("--port"), const_cast<char *>("8082"),
        const_cast<char *>("--no-default-args"),
        nullptr,
    };
    const int rc = posix_spawn(&pid, executable.c_str(), nullptr, nullptr, argv, environ);
    if (rc != 0) {
        err = "could not start bundled whisper-server: " + std::string(strerror(rc));
        return false;
    }
    LOG_INF("starting bundled whisper-server on 127.0.0.1:%d\n", kEmbeddedAsrPort);

    httplib::Client health("127.0.0.1", kEmbeddedAsrPort);
    health.set_connection_timeout(1, 0);
    health.set_read_timeout(1, 0);
    for (int i = 0; i < 360; ++i) {
        int status = 0;
        const pid_t exited = waitpid(pid, &status, WNOHANG);
        if (exited == pid) {
            pid = -1;
            err = "bundled whisper-server exited before becoming ready";
            return false;
        }
        auto response = health.Get("/health");
        if (response && response->status == 200) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    err = "timed out waiting for bundled whisper-server to load its model";
    kill(pid, SIGTERM);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    pid = -1;
    return false;
}

void stop_embedded_asr(pid_t & pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    pid = -1;
}

void on_signal(int) {
    if (httplib::Server * s = g_server.load()) {
        s->stop();
    }
}

void reply_error(httplib::Response & res, int status, const std::string & message) {
    const char * type = status >= 500 ? "server_error" : "invalid_request_error";
    nlohmann::json j = {{"error", {{"message", message}, {"type", type}}}};
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

int run_server(tts_engine & e, tutor_engine * tutor, const std::string & host, int port,
               const std::string & asr_url) {
    httplib::Server svr;
    ttsfile::job_queue<tts_job> queue(kMaxQueuedJobs);

    // Speech-to-speech accepts uploaded audio. Keep the upload bounded while
    // allowing ordinary recordings (up to 25 MiB).
    svr.set_payload_max_length(25 * 1024 * 1024);

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        ttsfile::speech_request sr;
        const std::string perr = ttsfile::parse_speech_request(req.body, sr);
        if (!perr.empty()) {
            reply_error(res, 400, perr);
            return;
        }

        auto job  = std::make_shared<tts_job>();
        job->text = sr.input;
        job->lang = sr.lang.empty() ? e.params->tts_lang : sr.lang;
        const bool wav = sr.format == "wav";

        if (sr.stream) {
            auto ch      = std::make_shared<ttsfile::audio_channel>();
            job->channel = ch;
            if (!queue.push(job)) {
                reply_error(res, 503, "server is busy or shutting down; retry later");
                return;
            }

            // Wait for the first audio (or a failure) before committing to a
            // 200: after the first byte the status line can no longer change.
            ch->wait_ready();
            if (ch->failed_before_audio()) {
                LOG_ERR("synthesis failed: %s\n", ch->error().c_str());
                reply_error(res, 500, ch->error());
                return;
            }

            ttsfile::stream_audio_response(res, ch, wav, (uint32_t) e.sample_rate);
            return;
        }

        auto fut = job->done.get_future();
        if (!queue.push(job)) {
            reply_error(res, 503, "server is busy or shutting down; retry later");
            return;
        }

        const job_outcome out = fut.get();  // main thread fulfils this
        if (!out.ok) {
            LOG_ERR("synthesis failed: %s\n", out.err.c_str());
            reply_error(res, 500, out.err);
            return;
        }
        if (wav) {
            res.set_content(out.result.wav, "audio/wav");
        } else {
            // the helper's WAV is a fixed 44-byte header followed by s16le samples
            res.set_header("X-Sample-Rate", std::to_string(out.result.sample_rate));
            res.set_header("X-Sample-Format", "s16le");
            res.set_header("X-Channels", "1");
            res.set_content(out.result.wav.substr(44), "audio/pcm");
        }
    });

    // First speech-to-speech iteration: transcribe an uploaded recording
    // with a local whisper.cpp server, generate a concise tutor reply, then
    // speak it with the already-loaded TTS model.
    svr.Post("/v1/audio/speech-to-speech", [&](const httplib::Request & req, httplib::Response & res) {
        if (!tutor) {
            reply_error(res, 503, "speech tutor requires --chat-model");
            return;
        }
        if (asr_url.empty()) {
            reply_error(res, 503, "speech-to-speech requires --asr-model or --asr-url");
            return;
        }
        if (!req.form.has_file("file")) {
            reply_error(res, 400, "multipart audio upload must include a file field named \"file\"");
            return;
        }
        const auto audio = req.form.get_file("file");
        if (audio.content.empty()) {
            reply_error(res, 400, "uploaded audio is empty");
            return;
        }

        httplib::Client asr(asr_url);
        asr.set_connection_timeout(5, 0);
        asr.set_read_timeout(300, 0);
        httplib::UploadFormDataItems fields = {
            {"file", audio.content, audio.filename.empty() ? "speech.wav" : audio.filename,
             audio.content_type.empty() ? "application/octet-stream" : audio.content_type},
            {"response_format", "verbose_json", "", ""},
            {"no_language_probabilities", "true", "", ""},
        };
        std::string lang;
        if (req.form.has_field("language")) {
            lang = req.form.get_field("language");
        } else if (req.has_param("language")) {
            lang = req.get_param_value("language");
        }
        if (!lang.empty()) {
            if (lang.size() != 2 || !std::isalpha((unsigned char) lang[0]) ||
                !std::isalpha((unsigned char) lang[1])) {
                reply_error(res, 400, "language must be a two-letter ISO 639-1 code such as \"en\"");
                return;
            }
            for (char & c : lang) c = (char) std::tolower((unsigned char) c);
            fields.push_back({"language", lang, "", ""});
        }
        auto asr_res = asr.Post("/inference", fields);
        if (!asr_res) {
            reply_error(res, 502, "could not reach the configured whisper-server");
            return;
        }
        if (asr_res->status < 200 || asr_res->status >= 300) {
            reply_error(res, 502, "whisper-server returned HTTP " + std::to_string(asr_res->status));
            return;
        }
        nlohmann::json transcript = nlohmann::json::parse(asr_res->body, nullptr, false);
        if (transcript.is_discarded() || !transcript.is_object() || !transcript.contains("text") ||
            !transcript["text"].is_string()) {
            reply_error(res, 502, "whisper-server returned an invalid JSON transcription");
            return;
        }
        const std::string spoken_text = ttsfile::detail::trim(transcript["text"].get<std::string>());
        if (spoken_text.empty()) {
            reply_error(res, 422, "no speech was recognized in the uploaded audio");
            return;
        }
        if (spoken_text.size() > 1000) {
            reply_error(res, 413, "recognized speech is too long (max 1000 bytes per turn)");
            return;
        }

        std::string session_id;
        if (req.form.has_field("session_id")) {
            session_id = req.form.get_field("session_id");
        } else if (req.has_param("session_id")) {
            session_id = req.get_param_value("session_id");
        }
        session_id = ttsfile::detail::trim(session_id);
        if (session_id.size() > 128) {
            reply_error(res, 400, "session_id must be at most 128 bytes");
            return;
        }

        auto job = std::make_shared<tts_job>();
        job->text = spoken_text;
        job->lang = lang.empty() ? tts_language_from_asr(transcript) : lang;
        if (job->lang.empty()) job->lang = e.params->tts_lang;
        job->session_id = session_id;
        job->tutor = true;
        auto fut = job->done.get_future();
        if (!queue.push(job)) {
            reply_error(res, 503, "server is busy or shutting down; retry later");
            return;
        }
        const job_outcome out = fut.get();
        if (!out.ok) {
            LOG_ERR("speech-to-speech synthesis failed: %s\n", out.err.c_str());
            reply_error(res, 500, out.err);
            return;
        }
        res.set_content(out.result.wav, "audio/wav");
    });

    g_server.store(&svr);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    // HTTP runs on its own thread; this (main) thread does the synthesis.
    // listen() returns false only if it could not bind/listen; a clean stop()
    // from the signal handler makes it return true. Either way, close the
    // queue so the loop below ends.
    bool listened = false;
    std::thread http_thread([&] {
        LOG_INF("ttsfile listening on http://%s:%d (POST /v1/audio/speech, GET /health)\n",
                host.c_str(), port);
        listened = svr.listen(host, port);
        queue.close();
    });

    while (auto job = queue.pop()) {
        run_job(e, tutor, *job);
    }

    http_thread.join();
    g_server.store(nullptr);
    if (!listened) {
        LOG_ERR("failed to listen on %s:%d (port in use?)\n", host.c_str(), port);
        return 1;
    }
    return 0;
}

bool write_file(const std::string & path, const std::string & data) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    return fclose(f) == 0 && ok;
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef COSMOCC
    ShowCrashReports();
    // merge default arguments embedded at /zip/.args (if any) with argv
    argc = cosmo_args("/zip/.args", &argv);
#endif

    // wrapper-owned flags, consumed before upstream's parser sees them
    if (ttsfile::take_flag(argc, argv, "--version")) {
        puts("ttsfile " TTSFILE_VERSION_STRING);
        return 0;
    }
    const bool  server_mode = ttsfile::take_flag(argc, argv, "--server");
    std::string host        = "127.0.0.1";
    std::string port_str    = "8081";
    bool        missing     = false;
    ttsfile::take_value(argc, argv, "--host", &host, &missing);
    ttsfile::take_value(argc, argv, "--port", &port_str, &missing);
    std::string chunk_str = "8";
    ttsfile::take_value(argc, argv, "--stream-chunk-frames", &chunk_str, &missing);
    std::string asr_url;
    ttsfile::take_value(argc, argv, "--asr-url", &asr_url, &missing);
    std::string asr_model_path;
    ttsfile::take_value(argc, argv, "--asr-model", &asr_model_path, &missing);
    std::string asr_executable = "/zip/whisper-server";
    ttsfile::take_value(argc, argv, "--asr-executable", &asr_executable, &missing);
    std::string chat_model_path;
    ttsfile::take_value(argc, argv, "--chat-model", &chat_model_path, &missing);
    if (missing) {
        fprintf(stderr, "error: server, ASR, and chat model options require a value\n");
        return 1;
    }
    if (!asr_url.empty() && !asr_model_path.empty()) {
        fprintf(stderr, "error: use only one of --asr-url and --asr-model\n");
        return 1;
    }
    char * chunk_end = nullptr;
    const long chunk_frames = strtol(chunk_str.c_str(), &chunk_end, 10);
    if (chunk_str.empty() || *chunk_end != '\0' || chunk_frames < 1 || chunk_frames > 512) {
        fprintf(stderr, "error: invalid --stream-chunk-frames '%s' (use 1-512)\n", chunk_str.c_str());
        return 1;
    }
    char * port_end = nullptr;
    const long port = strtol(port_str.c_str(), &port_end, 10);
    if (port_str.empty() || *port_end != '\0' || port < 1 || port > 65535) {
        fprintf(stderr, "error: invalid --port '%s'\n", port_str.c_str());
        return 1;
    }

#ifdef COSMOCC
    // Trigger llamafile's dynamic loading of GPU backends (Metal/CUDA/...),
    // exactly as llamafile/main.cpp does before it loads a model. Without
    // this call the backends are never brought in and inference silently
    // runs on the CPU, whatever -ngl says. GPU work must stay on this
    // (main) thread: run_server() already does all synthesis here.
    // Own cache directory (~/.ttsfile/v/<version>/). The Metal dylib cache is
    // keyed only by version number: if it were shared with a regular llamafile,
    // ttsfile could silently reuse a dylib built from a different ggml tree.
    llamafile_set_app_name("ttsfile");
    const bool have_gpu = llamafile_has_gpu();
#endif

    common_params params;
    common_init();
#ifdef COSMOCC
    // llamafile_has_gpu() returns false without printing anything for several
    // reasons (not Apple silicon, --gpu disabled, kernels failed to build, no
    // embedded Metal sources), so say what happened.
    LOG_INF("llamafile GPU backend: %s (gpu mode: %s)\n",
            have_gpu ? "loaded" : "NOT available, running on CPU", llamafile_describe_gpu());
#endif
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (!server_mode && params.prompt.empty()) {
        LOG_ERR("no prompt provided: use -p \"text\" for one-shot mode, or --server\n");
        return 1;
    }
    if (params.mmproj.path.empty()) {
        LOG_ERR("no mmproj provided, use --mmproj (or -hf with a repo that has one)\n");
        return 1;
    }

    // hidden states are passed to the audio generation helper
    params.embedding = true;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    common_sampler * smpl = llama_init->sampler(0);
    if (!model || !lctx || !smpl) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = params.mmproj_use_gpu;
    mtmd_params.device  = params.mmproj_device;
    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mtmd_params));
    if (!mctx) {
        LOG_ERR("failed to load mmproj %s\n", params.mmproj.path.c_str());
        return 1;
    }
    if (mtmd_gen_audio_get_info(mctx.get()).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        LOG_ERR("mmproj does not support audio generation\n");
        return 1;
    }

    // optional reference voice, loaded once and reused for every request
    mtmd::bitmap_ptr speaker_bitmap;
    if (!params.tts_speaker_file.empty()) {
        auto wrapper = mtmd_helper_bitmap_init_from_file(mctx.get(), params.tts_speaker_file.c_str(), false,
                                                         mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            LOG_ERR("failed to load speaker file %s\n", params.tts_speaker_file.c_str());
            return 1;
        }
        speaker_bitmap.reset(wrapper.bitmap);
    }

    tts_engine engine;
    engine.params  = &params;
    engine.lctx    = lctx;
    engine.mctx    = mctx.get();
    engine.smpl    = smpl;
    engine.speaker = speaker_bitmap.get();
    engine.sample_rate = mtmd_gen_audio_get_info(mctx.get()).sample_rate;
    engine.stream_chunk_frames = (int) chunk_frames;

    std::unique_ptr<tutor_engine> tutor;
    common_params chat_params;
    if (!chat_model_path.empty()) {
        // Reuse the parsed inference settings (GPU layers, CPU threads, etc.)
        // but load an independent text model and context for tutor replies.
        chat_params = params;
        chat_params.model.path = chat_model_path;
        chat_params.model.url.clear();
        chat_params.model.hf_repo.clear();
        chat_params.model.hf_file.clear();
        chat_params.model.docker_repo.clear();
        chat_params.mmproj.path.clear();
        chat_params.mmproj.url.clear();
        chat_params.mmproj.hf_repo.clear();
        chat_params.mmproj.hf_file.clear();
        chat_params.embedding = false;
        chat_params.n_ctx = 4096;
        chat_params.n_batch = 4096;
        chat_params.n_ubatch = 512;
        chat_params.n_predict = 160;
        chat_params.prompt.clear();
        chat_params.out_file.clear();
        chat_params.tts_speaker_file.clear();
        tutor.reset(new tutor_engine);
        tutor->init = common_init_from_params(chat_params);
        if (!tutor->init || !tutor->init->model() || !tutor->init->context() || !tutor->init->sampler(0)) {
            LOG_ERR("failed to initialize tutor chat model %s\n", chat_model_path.c_str());
            return 1;
        }
    }

    int rc = 0;
    if (server_mode) {
        pid_t asr_pid = -1;
        if (!asr_model_path.empty()) {
            std::string err;
            if (!start_embedded_asr(asr_executable, asr_model_path, asr_pid, err)) {
                LOG_ERR("%s\n", err.c_str());
                return 1;
            }
            asr_url = "http://127.0.0.1:" + std::to_string(kEmbeddedAsrPort);
        }
        rc = run_server(engine, tutor.get(), host, (int) port, asr_url);
        stop_embedded_asr(asr_pid);
    } else {
        tts_result  result;
        std::string err;
        if (!synthesize(engine, params.prompt, params.tts_lang, result, &err)) {
            LOG_ERR("synthesis failed: %s\n", err.c_str());
            rc = 1;
        } else if (!write_file(params.out_file, result.wav)) {
            LOG_ERR("failed to write %s\n", params.out_file.c_str());
            rc = 1;
        } else {
            LOG_INF("wrote %s\n", params.out_file.c_str());
        }
    }

    llama_backend_free();
    return rc;
}
