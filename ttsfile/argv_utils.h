// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// argv helpers for the flags the ttsfile wrapper owns. Upstream's parser
// rejects options it does not know (--server/--host/--port are only
// registered for the llama-server example), so these are removed from argv
// before it runs. Header-only and dependency-free so they can be unit tested;
// see argv_utils_test.cpp.
//
// All helpers keep argv NULL-terminated (argv[argc] == nullptr) and update
// argc, matching what main() and upstream's parser expect.

#pragma once

#include <cstring>
#include <string>

namespace ttsfile {

// Removes the first occurrence of `flag` and returns true if it was present.
inline bool take_flag(int & argc, char ** argv, const char * flag) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], flag)) {
            for (int j = i; j < argc; ++j) {
                argv[j] = argv[j + 1];  // also shifts the NULL terminator
            }
            --argc;
            return true;
        }
    }
    return false;
}

// Removes "--flag VALUE" or "--flag=VALUE", returning true with *value set.
// Returns false if the flag is absent, or present without a value (in which
// case *missing is set and argv is left untouched).
inline bool take_value(int & argc, char ** argv, const char * flag,
                       std::string * value, bool * missing) {
    const size_t n = strlen(flag);
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], flag)) {
            if (i + 1 >= argc) {
                *missing = true;
                return false;
            }
            *value = argv[i + 1];
            for (int j = i; j + 2 <= argc; ++j) {
                argv[j] = argv[j + 2];  // also shifts the NULL terminator
            }
            argc -= 2;
            return true;
        }
        if (!strncmp(argv[i], flag, n) && argv[i][n] == '=') {
            *value = argv[i] + n + 1;
            for (int j = i; j + 1 <= argc; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            return true;
        }
    }
    return false;
}

}  // namespace ttsfile
