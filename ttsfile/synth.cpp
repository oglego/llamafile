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
// The synthesis loop is deliberately a close copy of llama.cpp/tools/tts/
// tts.cpp so that behaviour matches llama-tts; keep the two in step when
// bumping the llama.cpp submodule.
//

#include "synth.h"

#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"

#include <string>

namespace ttsfile {

bool synthesize(tts_engine & e, const std::string & text, const std::string & lang,
                tts_result & res, std::string * err, const stream_sink * sink) {
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

}  // namespace ttsfile
