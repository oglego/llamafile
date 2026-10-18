// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Unit tests for ttsfile/speech_request.h. No model required.
//
//   g++ -std=c++17 -iquote ttsfile -isystem llama.cpp/vendor ttsfile/speech_request_test.cpp -o /tmp/srt && /tmp/srt

#include "speech_request.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static std::string parse(const std::string & body, ttsfile::speech_request & r,
                         size_t max = ttsfile::kMaxInputBytes) {
    return ttsfile::parse_speech_request(body, r, max);
}

int main() {
    ttsfile::speech_request r;

    // minimal valid request
    CHECK(parse(R"({"input":"Hello world"})", r).empty());
    CHECK(r.input == "Hello world");
    CHECK(r.lang.empty());

    // whitespace is trimmed; language is normalised to lower case
    r = {};
    CHECK(parse("{\"input\":\"  Hallo Welt \\n\",\"language\":\"DE\"}", r).empty());
    CHECK(r.input == "Hallo Welt");
    CHECK(r.lang == "de");

    // OpenAI-style extras are accepted and ignored; explicit wav is fine
    r = {};
    CHECK(parse(R"({"model":"x","voice":"alloy","input":"Hola","response_format":"wav","language":"es"})", r).empty());
    CHECK(r.input == "Hola" && r.lang == "es");

    // null optionals behave like absent ones
    r = {};
    CHECK(parse(R"({"input":"Hi","language":null,"response_format":null,"stream":null})", r).empty());
    CHECK(r.lang.empty() && r.format == "wav" && !r.stream);

    // defaults
    r = {};
    CHECK(parse(R"({"input":"Hi"})", r).empty());
    CHECK(r.format == "wav" && !r.stream);

    // pcm and streaming
    r = {};
    CHECK(parse(R"({"input":"Hi","response_format":"pcm","stream":true})", r).empty());
    CHECK(r.format == "pcm" && r.stream);
    r = {};
    CHECK(parse(R"({"input":"Hi","stream":true})", r).empty());
    CHECK(r.format == "wav" && r.stream);
    r = {};
    CHECK(parse(R"({"input":"Hi","stream":false,"response_format":"wav"})", r).empty());
    CHECK(!r.stream);

    // unicode input passes through untouched
    r = {};
    CHECK(parse(R"({"input":"Grüße aus München"})", r).empty());
    CHECK(r.input == "Grüße aus München");

    // errors: each must fail and must leave `out` untouched
    const char * bad[] = {
        "",                                          // empty body
        "not json",                                  // malformed
        "[]",                                        // not an object
        "\"just a string\"",                         // not an object
        "{}",                                        // missing input
        R"({"input":42})",                           // wrong type
        R"({"input":""})",                           // empty
        R"({"input":"   \t\n "})",                   // whitespace only
        R"({"input":"x","response_format":"mp3"})",  // unsupported format
        R"({"input":"x","response_format":1})",      // wrong type
        R"({"input":"x","response_format":"WAV"})",  // case-sensitive
        R"({"input":"x","stream":"yes"})",           // not a boolean
        R"({"input":"x","stream":1})",               // not a boolean
        R"({"input":"x","language":"english"})",     // not ISO 639-1
        R"({"input":"x","language":"e"})",           // too short
        R"({"input":"x","language":"e1"})",          // non-alpha
        R"({"input":"x","language":5})",             // wrong type
    };
    for (const char * body : bad) {
        ttsfile::speech_request sentinel;
        sentinel.input = "sentinel";
        sentinel.lang  = "xx";
        const std::string err = parse(body, sentinel);
        if (err.empty()) {
            fprintf(stderr, "expected failure for body: %s\n", body);
            ++g_failures;
        }
        CHECK(sentinel.input == "sentinel" && sentinel.lang == "xx");
    }

    // length limit counts bytes of the trimmed text
    r = {};
    CHECK(parse("{\"input\":\"" + std::string(10, 'a') + "\"}", r, 10).empty());
    CHECK(!parse("{\"input\":\"" + std::string(11, 'a') + "\"}", r, 10).empty());
    CHECK(parse("{\"input\":\"  " + std::string(10, 'a') + "  \"}", r, 10).empty());

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    puts("speech_request_test: all checks passed");
    return 0;
}
