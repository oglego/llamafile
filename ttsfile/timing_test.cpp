// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Unit tests for ttsfile/timing.h. No model required.
//
//   g++ -std=c++17 -iquote ttsfile ttsfile/timing_test.cpp -o /tmp/tt && /tmp/tt

#include "timing.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

static int g_failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

static bool has(const std::string & s, const char * needle) { return s.find(needle) != std::string::npos; }

int main() {
    ttsfile::turn_timing t;
    t.asr_s = 1.82; t.queue_s = 0.0003; t.chat_s = 2.301; t.chat_tokens = 41;
    t.tts_s = 5.1; t.audio_s = 4.4; t.total_s = 9.3;

    // Server-Timing: milliseconds, one decimal, comma separated, chat has a desc
    const std::string h = ttsfile::server_timing_header(t);
    CHECK(has(h, "asr;dur=1820.0"));
    CHECK(has(h, "queue;dur=0.3"));
    CHECK(has(h, "chat;dur=2301.0;desc=\"41 tokens\""));
    CHECK(has(h, "tts;dur=5100.0"));
    CHECK(has(h, "total;dur=9300.0"));
    CHECK(h.find("asr") < h.find("queue") && h.find("queue") < h.find("chat") &&
          h.find("chat") < h.find("tts") && h.find("tts") < h.find("total"));
    CHECK(!has(h, "\n"));

    // summary carries the derived numbers
    const std::string s = ttsfile::timing_summary(t);
    CHECK(has(s, "asr 1.82s"));
    CHECK(has(s, "41 tokens, 17.8 tok/s"));          // 41 / 2.301
    CHECK(has(s, "tts 5.10s for 4.40s of audio (0.86x real time)"));
    CHECK(has(s, "total 9.30s"));

    // stages that did not run: no desc, no ratios, never inf/nan
    ttsfile::turn_timing z;
    const std::string hz = ttsfile::server_timing_header(z);
    CHECK(has(hz, "chat;dur=0.0") && !has(hz, "desc"));
    const std::string sz = ttsfile::timing_summary(z);
    CHECK(!has(sz, "inf") && !has(sz, "nan") && !has(sz, "tok/s") && !has(sz, "real time"));
    ttsfile::turn_timing partial;
    partial.chat_tokens = 10;  // tokens but zero time: must not divide by zero
    CHECK(!has(ttsfile::timing_summary(partial), "inf"));

    // stopwatch
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    const double dt = ttsfile::seconds_since(t0);
    CHECK(dt >= 0.05 && dt < 1.0);

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    puts("timing_test: all checks passed");
    return 0;
}
