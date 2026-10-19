// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Per-stage timing for one speech-to-speech turn, formatted for the log and
// as a standard Server-Timing HTTP header. Header-only and model-free; see
// timing_test.cpp.

#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace ttsfile {

inline double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// All durations in seconds. A stage that did not run stays 0.
struct turn_timing {
    double asr_s       = 0.0;  // speech recognition (HTTP call to whisper-server)
    double queue_s     = 0.0;  // waiting for the main thread to pick the job up
    double chat_s      = 0.0;  // chat model generating the tutor's reply
    int    chat_tokens = 0;    // tokens generated for that reply
    double tts_s       = 0.0;  // speech synthesis
    double audio_s     = 0.0;  // length of the audio that synthesis produced
    double total_s     = 0.0;  // whole request, start to finish
};

// "asr;dur=1820.0, queue;dur=0.3, chat;dur=2301.0;desc=\"41 tokens\", ..."
// (durations in milliseconds, per the Server-Timing spec)
inline std::string server_timing_header(const turn_timing & t) {
    char buf[96];
    std::string h;
    auto add = [&](const char * name, double seconds, const std::string & desc = "") {
        std::snprintf(buf, sizeof(buf), "%s;dur=%.1f", name, seconds * 1000.0);
        if (!h.empty()) {
            h += ", ";
        }
        h += buf;
        if (!desc.empty()) {
            h += ";desc=\"" + desc + "\"";
        }
    };
    add("asr", t.asr_s);
    add("queue", t.queue_s);
    add("chat", t.chat_s, t.chat_tokens > 0 ? std::to_string(t.chat_tokens) + " tokens" : "");
    add("tts", t.tts_s);
    add("total", t.total_s);
    return h;
}

// One human-readable line for the log. Ratios are omitted rather than printed
// as inf/nan when a stage took no time.
inline std::string timing_summary(const turn_timing & t) {
    char buf[320];
    std::string chat;
    if (t.chat_tokens > 0 && t.chat_s > 0) {
        std::snprintf(buf, sizeof(buf), "chat %.2fs (%d tokens, %.1f tok/s)", t.chat_s, t.chat_tokens,
                      t.chat_tokens / t.chat_s);
        chat = buf;
    } else {
        std::snprintf(buf, sizeof(buf), "chat %.2fs", t.chat_s);
        chat = buf;
    }
    std::string tts;
    if (t.audio_s > 0 && t.tts_s > 0) {
        std::snprintf(buf, sizeof(buf), "tts %.2fs for %.2fs of audio (%.2fx real time)", t.tts_s, t.audio_s,
                      t.audio_s / t.tts_s);
        tts = buf;
    } else {
        std::snprintf(buf, sizeof(buf), "tts %.2fs", t.tts_s);
        tts = buf;
    }
    std::snprintf(buf, sizeof(buf), "turn: asr %.2fs | queued %.2fs | ", t.asr_s, t.queue_s);
    return std::string(buf) + chat + " | " + tts + " | " + [&] {
        char b2[48];
        std::snprintf(b2, sizeof(b2), "total %.2fs", t.total_s);
        return std::string(b2);
    }();
}

}  // namespace ttsfile
