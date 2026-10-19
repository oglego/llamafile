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
// The speech tutor's chat model and conversation history.
//

#pragma once

#include "common.h"

#include <deque>
#include <string>
#include <unordered_map>

namespace ttsfile {

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

// Formats the conversation so far plus `text` with the chat model's template,
// generates a short reply and records the exchange under `session_id` (if
// non-empty). Returns false with a message in *err on failure. Runs on the
// calling thread; callers must serialize access to the engine.
bool generate_tutor_reply(tutor_engine & e, const std::string & text, const std::string & lang,
                          const std::string & session_id, std::string & reply, std::string * err);

}  // namespace ttsfile
