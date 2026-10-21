// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Header-only helpers for ttsfile's default arguments. Kept free of
// cosmopolitan dependencies so they can be unit-tested with a host compiler.

#pragma once

#include <stdlib.h>
#include <string.h>

namespace ttsfile {

// True if a context size was already chosen: on the command line, in a bundled
// /zip/.args (cosmo_args merges those into argv before we look), or through
// the LLAMA_ARG_CTX_SIZE environment variable that llama.cpp also honors.
inline bool has_ctx_size(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (!strcmp(a, "-c") || !strcmp(a, "--ctx-size") ||
            !strncmp(a, "--ctx-size=", 11)) {
            return true;
        }
    }
    const char * env = getenv("LLAMA_ARG_CTX_SIZE");
    return env && *env;
}

// Returns argv with "-c <n_ctx>" appended unless a context size is already
// set. The new array is NULL-terminated and lives for the rest of the process.
// On allocation failure the original argv is returned unchanged.
inline char ** with_default_ctx(int * argc, char ** argv, const char * n_ctx) {
    if (has_ctx_size(*argc, argv)) {
        return argv;
    }
    char ** out = (char **) malloc((size_t) (*argc + 3) * sizeof(char *));
    if (!out) {
        return argv;
    }
    memcpy(out, argv, (size_t) *argc * sizeof(char *));
    out[(*argc)++] = (char *) "-c";
    out[(*argc)++] = (char *) n_ctx;
    out[*argc] = nullptr;
    return out;
}

}  // namespace ttsfile
