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
// Starts and stops a bundled whisper-server child process for the tutor.
//

#pragma once

#include <string>

#include <sys/types.h>

namespace ttsfile {

constexpr int kEmbeddedAsrPort = 8082;

// Spawns `executable -m model_path` on 127.0.0.1:kEmbeddedAsrPort and waits
// until its /health answers. On failure returns false with *err set and pid = -1.
bool start_embedded_asr(const std::string & executable, const std::string & model_path,
                        pid_t & pid, std::string & err);

// SIGTERMs and reaps the child, if any. Safe to call with pid <= 0.
void stop_embedded_asr(pid_t & pid);

}  // namespace ttsfile
