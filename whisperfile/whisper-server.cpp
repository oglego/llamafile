// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
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
#include <string.h>

#include "llamafile/llamafile.h"
#include "llamafile/version.h"

// Defined in patched server.cpp
int whisper_server_main(int argc, char ** argv);

int main(int argc, char ** argv) {
    llamafile_check_cpu();
    ShowCrashReports();

    // Handle --version flag
    if (llamafile_has(argv, "--version")) {
        puts("whisper-server v" LLAMAFILE_VERSION_STRING);
        return 0;
    }

    // When launched by a bundled speech application, its /zip/.args belongs
    // to the parent program and contains unrelated options. Let that parent
    // suppress this wrapper's defaults and pass the server arguments directly.
    bool load_default_args = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-default-args")) {
            for (int j = i; j < argc; ++j) argv[j] = argv[j + 1];
            --argc;
            load_default_args = false;
            break;
        }
    }

    // Load default arguments from embedded .args file (for packaged whisperfiles).
    if (load_default_args) argc = cosmo_args("/zip/.args", &argv);

    return whisper_server_main(argc, argv);
}
