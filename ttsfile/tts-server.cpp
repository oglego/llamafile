// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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


// ttsfile --server: keeps the TTS model loaded and serves an
// OpenAI-compatible POST /v1/audio/speech endpoint.
//
// The generation loop mirrors llama.cpp/tools/tts/tts.cpp. Keep the two in
// sync when upstream changes.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct server_opts {
    std::string host       = "127.0.0.1";
    int         port       = 8080;
    std::string voices_dir;            // optional: <dir>/<voice>.wav
    std::string api_key;               // optional: require "Authorization: Bearer <key>"
    size_t      max_input  = 4096;     // max characters of input text per request
};

enum class out_fmt { WAV, PCM };

struct speech_req {
    std::string text;
    std::string voice;
    std::string lang;
    out_fmt     fmt    = out_fmt::WAV;
    bool        stream = false;
};

// Strips the server-only flags out of argv so the rest can go through the
// regular llama.cpp TTS argument parser, which doesn't know about them.
static bool extract_server_opts(int & argc, char ** argv, server_opts & so) {
    int out = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--host") {
            const char * v = need("--host");
            if (!v) return false;
            so.host = v;
        } else if (a == "--port") {
            const char * v = need("--port");
            if (!v) return false;
            so.port = atoi(v);
        } else if (a == "--voices-dir") {
            const char * v = need("--voices-dir");
            if (!v) return false;
            so.voices_dir = v;
        } else if (a == "--api-key") {
            const char * v = need("--api-key");
            if (!v) return false;
            so.api_key = v;
        } else if (a == "--max-input-chars") {
            const char * v = need("--max-input-chars");
            if (!v) return false;
            so.max_input = (size_t) atoll(v);
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return true;
}

static bool valid_voice_name(const std::string & s) {
    if (s.empty() || s.size() > 64) {
        return false;
    }
    for (char c : s) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

// float [-1,1] -> int16 little-endian
static void append_s16le(std::string & out, const float * pcm, size_t n) {
    out.reserve(out.size() + n * 2);
    for (size_t i = 0; i < n; i++) {
        float v = pcm[i];
        v = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
        const int16_t s = (int16_t) (v * 32767.0f);
        out.push_back((char) (s & 0xff));
        out.push_back((char) ((s >> 8) & 0xff));
    }
}

// Owns the model and serializes access to it: there is a single llama_context,
// so one request is generated at a time.
class tts_engine {
public:
    bool init(common_params & params, const server_opts & so) {
        so_     = so;
        params_ = &params;

        params.embedding = true; // hidden states are needed by the audio generation helper

        llama_backend_init();
        llama_numa_init(params.numa);

        init_ = common_init_from_params(params);
        model_ = init_->model();
        lctx_  = init_->context();
        smpl_  = init_->sampler(0);
        if (!model_ || !lctx_) {
            LOG_ERR("failed to init model/context\n");
            return false;
        }

        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = params.mmproj_use_gpu;
        mp.device  = params.mmproj_device;
        mctx_.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model_, mp));
        if (!mctx_) {
            LOG_ERR("failed to load mmproj %s\n", params.mmproj.path.c_str());
            return false;
        }
        if (mtmd_gen_audio_get_info(mctx_.get()).type == MTMD_GEN_AUDIO_TYPE_NONE) {
            LOG_ERR("mmproj does not support audio generation\n");
            return false;
        }

        if (!params.tts_speaker_file.empty()) {
            if (!load_voice("", params.tts_speaker_file)) {
                return false;
            }
        }
        gen_.reset(new mtmd_helper::gen_audio(lctx_, mctx_.get()));
        return true;
    }

    // true if a voice name from a request can be satisfied
    bool resolve_voice(const std::string & name, std::string & err) {
        std::lock_guard<std::mutex> lock(mu_);
        if (so_.voices_dir.empty() || name.empty()) {
            return true; // fall back to the default speaker, if any
        }
        if (!valid_voice_name(name)) {
            err = "invalid voice name";
            return false;
        }
        if (voices_.count(name)) {
            return true;
        }
        if (!load_voice(name, so_.voices_dir + "/" + name + ".wav")) {
            err = "unknown voice: " + name;
            return false;
        }
        return true;
    }

    // Generates audio for one request. `sink` receives audio bytes as they are
    // produced when req.stream is set, or once at the end otherwise. It returns
    // false to abort (e.g. client went away). Returns 0 on success.
    int synthesize(const speech_req & req, const std::function<bool(const char *, size_t)> & sink) {
        std::lock_guard<std::mutex> lock(mu_);

        // Start every request from a clean slate. The generation helper resets
        // its own state in set_input(), but the backbone KV cache and sampler
        // history belong to us.
        llama_memory_clear(llama_get_memory(lctx_), true);
        common_sampler_reset(smpl_);

        mtmd_bitmap * speaker = nullptr;
        {
            auto it = voices_.find(so_.voices_dir.empty() ? std::string() : req.voice);
            if (it == voices_.end()) {
                it = voices_.find(""); // default speaker
            }
            if (it != voices_.end()) {
                speaker = it->second.get();
            }
        }

        const std::string & lang = req.lang.empty() ? params_->tts_lang : req.lang;

        mtmd_helper_gen_audio_inp inp{};
        inp.seq_id      = 0;
        inp.prompt      = req.text.c_str();
        inp.prompt_len  = req.text.size();
        inp.speaker_ref = speaker;
        inp.lang        = lang.c_str();
        inp.top_k       = params_->sampling.top_k;
        inp.top_p       = params_->sampling.top_p;
        inp.seed        = params_->sampling.seed;
        inp.out_type    = (req.fmt == out_fmt::WAV) ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV
                                                    : MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;

        if (gen_->set_input(&inp) != 0) {
            LOG_ERR("set_input failed\n");
            return 1;
        }
        for (;;) {
            const int32_t ret = gen_->step_prompt(params_->n_batch);
            if (ret < 0) {
                LOG_ERR("prompt processing failed\n");
                return 1;
            }
            if (ret == 0) {
                break;
            }
        }

        auto sample = [&]() -> llama_token {
            llama_token t = common_sampler_sample(smpl_, lctx_, -1);
            common_sampler_accept(smpl_, t, true);
            return t;
        };

        size_t emitted_samples = 0;
        size_t emitted_wav_bytes = 0;
        bool wav_header_sent = false;

        // The helper returns all audio generated so far, so streaming sends
        // only the portion not already written to the sink.
        auto emit = [&](bool final) -> bool {
            if (!req.stream && !final) {
                return true;
            }
            int32_t      sr = 0;
            const char * data = nullptr;
            size_t       len = 0;
            if (gen_->get_output(&sr, &data, &len, nullptr) != 0) {
                return false;
            }
            if (!req.stream) {
                if (req.fmt == out_fmt::PCM) {
                    std::string s16;
                    append_s16le(s16, (const float *) data, len / sizeof(float));
                    return sink(s16.data(), s16.size());
                }
                return sink(data, len);
            }
            if (req.fmt == out_fmt::PCM) {
                const size_t n_samples = len / sizeof(float);
                if (n_samples < emitted_samples) {
                    LOG_ERR("audio output shrank while streaming\n");
                    return false;
                }
                std::string s16;
                append_s16le(s16, (const float *) data + emitted_samples, n_samples - emitted_samples);
                emitted_samples = n_samples;
                return s16.empty() || sink(s16.data(), s16.size());
            }
            if (len < 44 || len - 44 < emitted_wav_bytes) {
                LOG_ERR("WAV output shrank while streaming\n");
                return false;
            }
            if (!wav_header_sent) {
                std::string header(data, 44);
                memset(&header[4], 0xff, 4);
                memset(&header[40], 0xff, 4);
                if (!sink(header.data(), header.size())) {
                    return false;
                }
                wav_header_sent = true;
            }
            const size_t n_audio_bytes = len - 44 - emitted_wav_bytes;
            const bool ok = n_audio_bytes == 0 ||
                            sink(data + 44 + emitted_wav_bytes, n_audio_bytes);
            emitted_wav_bytes = len - 44;
            return ok;
        };

        const int max_new = params_->n_predict > 0 ? params_->n_predict : 512;
        int           n_frames = 0;
        llama_token   sampled  = sample();
        const float * h_state  = llama_get_embeddings_ith(lctx_, -1);
        bool          stop     = false;
        bool          alive    = true;

        while (!stop && n_frames < max_new && alive) {
            const float * h_next = nullptr;
            if (gen_->step_gen(sampled, h_state, &h_next, &stop) != 0) {
                LOG_ERR("step_gen failed at frame %d\n", n_frames);
                return 1;
            }
            if (!h_next) {
                break;
            }
            n_frames++;
            h_state = h_next;
            sampled = sample();
            if (req.stream) {
                alive = emit(false);
            }
        }
        if (!alive) {
            return 2; // client disconnected
        }

        if (!emit(true)) {
            return 1;
        }
        LOG_INF("generated %d frames for %zu chars of input\n", n_frames, req.text.size());
        return 0;
    }

    std::string model_name() const {
        const std::string & p = params_->model.path;
        const size_t slash = p.find_last_of("/\\");
        return slash == std::string::npos ? p : p.substr(slash + 1);
    }

    void shutdown() {
        gen_.reset();
        voices_.clear();
        mctx_.reset();
        init_.reset();
        llama_backend_free();
    }

private:
    bool load_voice(const std::string & key, const std::string & path) {
        auto wrapper = mtmd_helper_bitmap_init_from_file(mctx_.get(), path.c_str(), false,
                                                         mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            LOG_ERR("failed to load speaker file %s\n", path.c_str());
            return false;
        }
        voices_[key].reset(wrapper.bitmap);
        return true;
    }

    server_opts    so_;
    common_params * params_ = nullptr;

    common_init_result_ptr init_;
    llama_model *   model_ = nullptr;
    llama_context * lctx_  = nullptr;
    common_sampler * smpl_ = nullptr;

    mtmd::context_ptr mctx_;
    std::map<std::string, mtmd::bitmap_ptr> voices_; // "" = default speaker
    std::unique_ptr<mtmd_helper::gen_audio> gen_;

    std::mutex mu_;
};

static void json_error(httplib::Response & res, int status, const std::string & msg) {
    json j = {{"error", {{"message", msg}, {"type", "invalid_request_error"}}}};
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

} // namespace

int tts_server_main(int argc, char ** argv) {
    server_opts so;
    if (!extract_server_opts(argc, argv, so)) {
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS)) {
        return 1;
    }
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        LOG_ERR("no mmproj provided, use --mmproj\n");
        return 1;
    }

    tts_engine engine;
    if (!engine.init(params, so)) {
        return 1;
    }

    httplib::Server svr;

    // optional bearer-token auth, applied to everything except /health
    svr.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        if (so.api_key.empty() || req.path == "/health") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        if (req.get_header_value("Authorization") != "Bearer " + so.api_key) {
            json_error(res, 401, "invalid api key");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {{"object", "list"},
                  {"data", json::array({{{"id", engine.model_name()}, {"object", "model"}, {"owned_by", "llamafile"}}})}};
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request & http_req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(http_req.body);
        } catch (const std::exception &) {
            json_error(res, 400, "request body is not valid JSON");
            return;
        }

        auto req = std::make_shared<speech_req>();
        try {
            req->text = body.at("input").get<std::string>();
        } catch (const std::exception &) {
            json_error(res, 400, "'input' (string) is required");
            return;
        }
        if (req->text.empty()) {
            json_error(res, 400, "'input' must not be empty");
            return;
        }
        if (req->text.size() > so.max_input) {
            json_error(res, 400, "'input' is longer than " + std::to_string(so.max_input) + " characters");
            return;
        }

        req->voice = body.value("voice", std::string());
        req->lang  = body.value("language", std::string()); // extension, not part of the OpenAI API

        const std::string fmt = body.value("response_format", std::string("wav"));
        if (fmt == "wav") {
            req->fmt = out_fmt::WAV;
        } else if (fmt == "pcm") {
            req->fmt = out_fmt::PCM; // 16-bit little-endian mono, like OpenAI's pcm
        } else {
            json_error(res, 400, "response_format '" + fmt + "' is not supported, use 'wav' or 'pcm'");
            return;
        }

        // OpenAI uses stream_format ("audio" | "sse"); we also accept a plain
        // "stream": true. Only raw audio streaming is implemented.
        const std::string stream_format = body.value("stream_format", std::string("audio"));
        if (stream_format != "audio") {
            json_error(res, 400, "only stream_format 'audio' is supported");
            return;
        }
        req->stream = body.value("stream", false);

        std::string err;
        if (!engine.resolve_voice(req->voice, err)) {
            json_error(res, 400, err);
            return;
        }

        const char * mime = req->fmt == out_fmt::WAV ? "audio/wav" : "audio/pcm";

        if (!req->stream) {
            std::string audio;
            const int rc = engine.synthesize(*req, [&](const char * d, size_t n) {
                audio.append(d, n);
                return true;
            });
            if (rc != 0) {
                json_error(res, 500, "speech generation failed");
                return;
            }
            res.set_content(audio, mime);
            return;
        }

        // Streaming: generation runs inside the chunked provider, which writes
        // audio to the client as each window of frames is decoded.
        res.set_chunked_content_provider(mime, [&engine, req](size_t, httplib::DataSink & sink) {
            engine.synthesize(*req, [&](const char * d, size_t n) { return sink.write(d, n); });
            sink.done();
            return true;
        });
    });

    LOG_INF("ttsfile server listening on http://%s:%d (POST /v1/audio/speech)\n", so.host.c_str(), so.port);
    if (!svr.listen(so.host.c_str(), so.port)) {
        LOG_ERR("failed to listen on %s:%d\n", so.host.c_str(), so.port);
        engine.shutdown();
        return 1;
    }
    engine.shutdown();
    return 0;
}
