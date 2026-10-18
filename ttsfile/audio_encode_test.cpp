// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Unit tests for ttsfile/audio_encode.h. No model required.
//
//   g++ -std=c++17 -iquote ttsfile ttsfile/audio_encode_test.cpp -o /tmp/aet && /tmp/aet

#include "audio_encode.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

static int16_t s16_at(const std::string & s, size_t i) {
    const uint16_t u = static_cast<uint8_t>(s[2 * i]) | (static_cast<uint16_t>(static_cast<uint8_t>(s[2 * i + 1])) << 8);
    return static_cast<int16_t>(u);
}

static uint32_t u32_at(const std::string & s, size_t off) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<uint32_t>(static_cast<uint8_t>(s[off + i])) << (8 * i);
    }
    return v;
}

int main() {
    // conversion matches llama.cpp's writer: clamp, scale by 32767, truncate
    {
        const float in[] = {0.0f, 1.0f, -1.0f, 2.5f, -2.5f, 0.5f, -0.5f, 1e-6f};
        std::string out;
        ttsfile::append_s16le(out, in, 8);
        CHECK(out.size() == 16);
        CHECK(s16_at(out, 0) == 0);
        CHECK(s16_at(out, 1) == 32767);
        CHECK(s16_at(out, 2) == -32767);
        CHECK(s16_at(out, 3) == 32767);    // clamped
        CHECK(s16_at(out, 4) == -32767);   // clamped
        CHECK(s16_at(out, 5) == 16383);    // 16383.5 truncates toward zero
        CHECK(s16_at(out, 6) == -16383);
        CHECK(s16_at(out, 7) == 0);
    }
    // little-endian byte order: 0x1234 -> 34 12
    {
        // 4660 / 32767 truncates back to 4660 when multiplied (checked below)
        const float in[] = {4660.0f / 32767.0f};
        std::string out;
        ttsfile::append_s16le(out, in, 1);
        const int16_t s = s16_at(out, 0);
        CHECK(s == 4660 || s == 4659);  // float rounding may land one below
        CHECK(static_cast<uint8_t>(out[1]) == ((s >> 8) & 0xff));
        CHECK(static_cast<uint8_t>(out[0]) == (s & 0xff));
    }
    // appending in pieces equals converting at once (what streaming relies on)
    {
        std::vector<float> in;
        for (int i = 0; i < 1000; ++i) {
            in.push_back(((i * 37) % 200 - 100) / 90.0f);  // includes values beyond +-1
        }
        std::string whole, pieces;
        ttsfile::append_s16le(whole, in.data(), in.size());
        size_t off = 0;
        for (size_t step : {1u, 7u, 64u, 300u, 628u}) {
            const size_t n = std::min(step, in.size() - off);
            ttsfile::append_s16le(pieces, in.data() + off, n);
            off += n;
        }
        CHECK(off == in.size());
        CHECK(whole == pieces);
    }
    // empty append is a no-op
    {
        std::string out = "x";
        ttsfile::append_s16le(out, nullptr, 0);
        CHECK(out == "x");
    }
    // streaming WAV header: 44 bytes, mono 16-bit PCM, unknown length
    {
        const std::string h = ttsfile::streaming_wav_header(24000);
        CHECK(h.size() == 44);
        CHECK(h.compare(0, 4, "RIFF") == 0);
        CHECK(u32_at(h, 4) == 0xFFFFFFFFu);
        CHECK(h.compare(8, 4, "WAVE") == 0);
        CHECK(h.compare(12, 4, "fmt ") == 0);
        CHECK(u32_at(h, 16) == 16);
        CHECK(static_cast<uint8_t>(h[20]) == 1 && static_cast<uint8_t>(h[21]) == 0);  // PCM
        CHECK(static_cast<uint8_t>(h[22]) == 1 && static_cast<uint8_t>(h[23]) == 0);  // mono
        CHECK(u32_at(h, 24) == 24000);
        CHECK(u32_at(h, 28) == 48000);
        CHECK(static_cast<uint8_t>(h[32]) == 2 && static_cast<uint8_t>(h[33]) == 0);
        CHECK(static_cast<uint8_t>(h[34]) == 16 && static_cast<uint8_t>(h[35]) == 0);
        CHECK(h.compare(36, 4, "data") == 0);
        CHECK(u32_at(h, 40) == 0xFFFFFFFFu);
    }
    // header is the same size as the one llama.cpp writes, so a streamed file
    // (header + chunks) lines up with a non-streamed one for byte comparison
    CHECK(ttsfile::streaming_wav_header(16000).size() == 44);

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    puts("audio_encode_test: all checks passed");
    return 0;
}
