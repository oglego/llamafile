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

#include <cosmo.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "llamafile/llamafile.h"

// Forward declaration - defined in llama.cpp/tools/tts/tts.cpp.
// ttsfile/BUILD.mk compiles that file with -Dmain=tts_cli_main so its
// main() becomes this function and can be called from here.
int tts_cli_main(int argc, char ** argv);

int main(int argc, char ** argv) {
    llamafile_check_cpu();
    ShowCrashReports();

    // Load default arguments from embedded .args file (for packaged ttsfiles)
    argc = cosmo_args("/zip/.args", &argv);

    // Without -c, llama.cpp sizes the context to the model's full training
    // window, and the KV cache can exhaust GPU memory before the first frames
    // are generated. Default to 2048 tokens unless the user chose a size.
    bool has_ctx = getenv("LLAMA_ARG_CTX_SIZE") != nullptr;
    for (int i = 1; i < argc; ++i) {
        has_ctx |= !strcmp(argv[i], "-c") || !strcmp(argv[i], "--ctx-size");
    }

    // Keep runtime-compiled GPU artifacts separate from llamafile's cache.
    llamafile_set_app_name("ttsfile");
    llamafile_has_gpu();

    static char ctx_flag[] = "-c";
    static char ctx_size[] = "2048";
    std::vector<char *> args(argv, argv + argc);
    if (!has_ctx) {
        args.push_back(ctx_flag);
        args.push_back(ctx_size);
    }
    args.push_back(nullptr);
    return tts_cli_main((int) args.size() - 1, args.data());
}
