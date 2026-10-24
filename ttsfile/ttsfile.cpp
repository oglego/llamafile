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

#include <cosmo.h>
#include <stdio.h>

#include "llamafile/llamafile.h"
#include "llamafile/version.h"

// Forward declaration - defined in llama.cpp/tools/tts/tts.cpp.
// ttsfile/BUILD.mk compiles that file with -Dmain=tts_cli_main so its
// main() becomes this function and can be called from here.
int tts_cli_main(int argc, char ** argv);

int main(int argc, char ** argv) {
    // CPU feature check and crash reports
    llamafile_check_cpu();
    ShowCrashReports();

    // Handle --version before anything else
    if (llamafile_has(argv, "--version")) {
        puts("ttsfile v" LLAMAFILE_VERSION_STRING);
        return 0;
    }

    // Load default arguments from embedded .args file (for packaged ttsfiles)
    argc = cosmo_args("/zip/.args", &argv);

    return tts_cli_main(argc, argv);
}
