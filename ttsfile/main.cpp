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
#include "job_queue.h"
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
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#ifdef COSMOCC
#include <cosmo.h>
#endif

#ifndef TTSFILE_VERSION_STRING
#define TTSFILE_VERSION_STRING "0.0.0-dev"
#endif

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
};

struct tts_result {
    std::string wav;
    int         n_frames    = 0;
    int32_t     sample_rate = 0;
    int64_t     n_samples   = 0;
    double      t_total_s   = 0.0;
};

// Synthesizes `text` and fills `res`. Returns false (with a message in *err)
// on failure. Not thread-safe: callers must serialize access to the engine.
bool synthesize(tts_engine & e, const std::string & text, const std::string & lang,
                tts_result & res, std::string * err) {
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
    inp.out_type    = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;

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
        sampled = sample_semantic_code();
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
// loaded the model (main), one job at a time. Running the model on httplib's
// worker threads is a bad idea: their stacks are small under cosmocc (a
// stack overflow there kills the process with a bare "bus error", because
// cosmo's crash reporter only works on the main thread), and Metal/ggml
// state is happiest when used from the thread that created it.
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
    std::promise<job_outcome>  done;
};

std::atomic<httplib::Server *> g_server{nullptr};

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

int run_server(tts_engine & e, const std::string & host, int port) {
    httplib::Server svr;
    ttsfile::job_queue<tts_job> queue(kMaxQueuedJobs);

    svr.set_payload_max_length(64 * 1024);  // requests are small JSON documents

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
        auto fut  = job->done.get_future();
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
        job_outcome out;
        out.ok = synthesize(e, job->text, job->lang, out.result, &out.err);
        job->done.set_value(std::move(out));
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
    if (missing) {
        fprintf(stderr, "error: --host/--port require a value\n");
        return 1;
    }
    char * port_end = nullptr;
    const long port = strtol(port_str.c_str(), &port_end, 10);
    if (port_str.empty() || *port_end != '\0' || port < 1 || port > 65535) {
        fprintf(stderr, "error: invalid --port '%s'\n", port_str.c_str());
        return 1;
    }

    common_params params;
    common_init();
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

    int rc = 0;
    if (server_mode) {
        rc = run_server(engine, host, (int) port);
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
