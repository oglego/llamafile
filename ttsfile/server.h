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
// The HTTP server: text-to-speech and speech-to-speech endpoints.
//

#pragma once

#include "synth.h"
#include "tutor.h"

#include <string>

namespace ttsfile {

// Serves POST /v1/audio/speech, POST /v1/audio/speech-to-speech and
// GET /health on host:port until SIGINT/SIGTERM. HTTP runs on its own thread;
// ALL synthesis and chat generation runs on the calling thread, one job at a
// time. `tutor` may be null (speech-to-speech then answers 503). Returns a
// process exit code.
int run_server(tts_engine & e, tutor_engine * tutor, const std::string & host, int port,
               const std::string & asr_url);

}  // namespace ttsfile
