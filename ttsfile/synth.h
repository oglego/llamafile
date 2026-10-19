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
// Text-to-speech synthesis on top of llama.cpp's libmtmd audio generation.
//

#pragma once

#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "sampling.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace ttsfile {

// everything loaded once, one request at a time
struct tts_engine {
    common_params      * params = nullptr;
    llama_context      * lctx   = nullptr;
    mtmd_context       * mctx   = nullptr;
    common_sampler     * smpl   = nullptr;
    mtmd_bitmap        * speaker = nullptr;  // optional reference voice, owned by caller
    int                  sample_rate = 0;    // Hz, from the model
    int                  stream_chunk_frames = 8;
};

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
                tts_result & res, std::string * err, const stream_sink * sink = nullptr);

}  // namespace ttsfile
