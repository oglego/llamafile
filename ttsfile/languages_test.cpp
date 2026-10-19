// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Unit tests for ttsfile/languages.h. No model required.
//
//   g++ -std=c++17 -iquote ttsfile -isystem llama.cpp/vendor ttsfile/languages_test.cpp -o /tmp/lt && /tmp/lt

#include "languages.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

using nlohmann::json;

int main() {
    // code -> name for the ten Qwen3-TTS languages
    const std::pair<const char *, const char *> codes[] = {
        {"zh", "Chinese"}, {"en", "English"}, {"de", "German"}, {"it", "Italian"},
        {"pt", "Portuguese"}, {"es", "Spanish"}, {"ja", "Japanese"}, {"ko", "Korean"},
        {"fr", "French"}, {"ru", "Russian"},
    };
    for (const auto & c : codes) {
        CHECK(ttsfile::tutor_language_name(c.first) == c.second);
    }
    // anything else passes through unchanged, so the chat prompt still names it
    CHECK(ttsfile::tutor_language_name("nl") == "nl");
    CHECK(ttsfile::tutor_language_name("") == "");

    // Whisper name -> TTS code, same ten languages
    for (const auto & c : codes) {
        std::string lower = c.second;
        for (char & ch : lower) ch = (char) std::tolower((unsigned char) ch);
        CHECK(ttsfile::tts_language_from_asr(json{{"language", lower}}) == c.first);
    }
    // case-insensitive
    CHECK(ttsfile::tts_language_from_asr(json{{"language", "German"}}) == "de");
    CHECK(ttsfile::tts_language_from_asr(json{{"language", "ENGLISH"}}) == "en");

    // no mapping -> empty string, so the caller falls back to --tts-lang
    CHECK(ttsfile::tts_language_from_asr(json{{"language", "dutch"}}).empty());
    CHECK(ttsfile::tts_language_from_asr(json{{"language", ""}}).empty());
    CHECK(ttsfile::tts_language_from_asr(json{{"text", "hello"}}).empty());   // key missing
    CHECK(ttsfile::tts_language_from_asr(json{{"language", 5}}).empty());     // wrong type
    CHECK(ttsfile::tts_language_from_asr(json{{"language", nullptr}}).empty());
    CHECK(ttsfile::tts_language_from_asr(json::array()).empty());             // not an object
    CHECK(ttsfile::tts_language_from_asr(json()).empty());                    // null

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    puts("languages_test: all checks passed");
    return 0;
}
