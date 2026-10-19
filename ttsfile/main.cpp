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

#include "argv_utils.h"
#include "asr_process.h"
#include "server.h"
#include "synth.h"
#include "tutor.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <sys/types.h>

#ifdef COSMOCC
#include <cosmo.h>
#include "llamafile/llamafile.h"
#endif

#ifndef TTSFILE_VERSION_STRING
#define TTSFILE_VERSION_STRING "0.0.0-dev"
#endif

using namespace ttsfile;

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
