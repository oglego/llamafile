// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// ttsfile entry point.
//
// Wraps llama.cpp's TTS tool (tools/tts/tts.cpp, built on libmtmd), whose
// main() is renamed to tts_cli_main() via -Dmain=tts_cli_main in
// llama.cpp.patches/llamafile-files/BUILD.mk, with the llamafile-style
// conveniences: a crash handler, GPU backend registration, and loading
// default arguments from the executable's own zip store at /zip/.args.
//
// That last bit is what makes a self-contained `foo.ttsfile` possible:
// zipalign the backbone GGUF, the mmproj GGUF and a speaker reference
// audio file into the executable, plus a .args file such as
//
//     -m
//     /zip/model.gguf
//     -mm
//     /zip/mmproj.gguf
//     --tts-speaker-file
//     /zip/voice.wav
//
// and it runs with only `-p "text to speak" -o out.wav`.

#include <cosmo.h>
#include <stdio.h>
#include <string.h>

#include "llamafile/llamafile.h"
#include "ttsfile/default_args.h"

// Defined in llama.cpp/tools/tts/tts.cpp (main renamed at compile time).
// C++ linkage on both sides.
int tts_cli_main(int argc, char ** argv);

// Default backbone context size. Upstream deliberately leaves -c at the model's
// full training length (a future TTS model may need more), which for Qwen3-TTS
// means a ~3.5 GiB KV cache (32768 cells) that can exhaust GPU memory on small
// machines (seen on an 8 GB Apple M2 with Metal). A TTS utterance needs only
// the prompt, any speaker reference, and one position per generated frame, so
// ttsfile defaults to 4096 and leaves the choice to the user: pass -c N (or set
// LLAMA_ARG_CTX_SIZE, or put -c in a bundled .args) to override.
#define TTSFILE_DEFAULT_CTX "4096"

#ifndef TTSFILE_VERSION_STRING
#define TTSFILE_VERSION_STRING "0.0.0-dev"
#endif

static bool has_flag(char ** argv, const char * flag) {
    for (char ** p = argv + 1; *p; ++p) {
        if (!strcmp(*p, flag)) {
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    // Symbolized backtraces on crash (cosmopolitan).
    ShowCrashReports();

    // Cached artifacts (the runtime-compiled Metal dylib, GPU DSOs) go under
    // ~/.ttsfile so they never collide with another llamafile product's.
    llamafile_set_app_name("ttsfile");

    // Answer --version before touching args or the zip store.
    if (has_flag(argv, "--version")) {
        puts("ttsfile " TTSFILE_VERSION_STRING);
        return 0;
    }

    // Merge default arguments embedded at /zip/.args (if present) with the
    // user's argv. No-op for a bare executable with no bundled .args.
    argc = cosmo_args("/zip/.args", &argv);

    // Add -c only if neither the user nor a bundled .args chose a size.
    argv = ttsfile::with_default_ctx(&argc, argv, TTSFILE_DEFAULT_CTX);

    // Triggers Metal backend registration on macOS ARM64 (no-op elsewhere).
    // CUDA/Vulkan are picked up through the DSO loaders, as in llama-server.
    llamafile_has_metal();

    return tts_cli_main(argc, argv);
}
