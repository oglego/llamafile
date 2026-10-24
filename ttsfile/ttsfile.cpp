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
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "llamafile/llamafile.h"
#include "llamafile/version.h"

// Forward declaration - defined in llama.cpp/tools/tts/tts.cpp.
// ttsfile/BUILD.mk compiles that file with -Dmain=tts_cli_main so its
// main() becomes this function and can be called from here.
int tts_cli_main(int argc, char ** argv);

static std::string read_all_stdin() {
    std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

static bool is_flag(const char * a, const char * x, const char * y = nullptr) {
    return !strcmp(a, x) || (y && !strcmp(a, y));
}

// Scriptability helpers for the CLI:
//   -p -            read the text to speak from stdin
//   (no -p at all)  if stdin is piped, read the text from it
//   -o -            write the WAV to stdout
// Rewrites argv into `storage`/`out` and returns the new argc.
static int apply_stdio_args(int argc, char ** argv, std::vector<std::string> & storage, std::vector<char *> & out) {
    bool has_prompt = false;
    bool no_run     = false; // --help / --version style invocations
    for (int i = 1; i < argc; i++) {
        if (is_flag(argv[i], "-p", "--prompt")) has_prompt = true;
        if (is_flag(argv[i], "-h", "--help") || is_flag(argv[i], "--version")) no_run = true;
    }

    storage.reserve(argc + 2);
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (i > 0 && i + 1 < argc) {
            if (is_flag(argv[i], "-p", "--prompt") && !strcmp(argv[i + 1], "-")) {
                storage.push_back(a);
                storage.push_back(read_all_stdin());
                i++;
                continue;
            }
            if (is_flag(argv[i], "-o", "--output") && !strcmp(argv[i + 1], "-")) {
                // Logs go to stderr, so stdout carries only the audio.
                storage.push_back(a);
                storage.push_back("/dev/stdout");
                i++;
                continue;
            }
        }
        storage.push_back(a);
    }

    if (!has_prompt && !no_run && !isatty(STDIN_FILENO)) {
        std::string text = read_all_stdin();
        if (!text.empty()) {
            storage.push_back("-p");
            storage.push_back(text);
        }
    }

    for (auto & s : storage) out.push_back(&s[0]);
    out.push_back(nullptr);
    return (int) storage.size();
}

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

    // Register llamafile's dynamically loaded GPU backends before llama.cpp
    // initializes its backend registry. Help should stay fast and usable on
    // machines without a working GPU toolchain.
    if (!llamafile_has(argv, "-h") && !llamafile_has(argv, "--help")) {
        llamafile_has_gpu();
    }

    std::vector<std::string> storage;
    std::vector<char *> args;
    argc = apply_stdio_args(argc, argv, storage, args);
    return tts_cli_main(argc, args.data());
}
