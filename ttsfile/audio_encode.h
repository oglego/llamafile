// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Sample conversion and WAV framing for ttsfile's streaming output.
// Header-only, no dependencies; see audio_encode_test.cpp.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ttsfile {

// Appends `n` float samples (nominally -1..1) as little-endian signed 16-bit
// PCM. Uses the same conversion as llama.cpp's WAV writer (clamp, scale by
// 32767, truncate toward zero) so streamed and non-streamed output match
// sample for sample.
inline void append_s16le(std::string & out, const float * pcm, size_t n) {
    out.reserve(out.size() + n * 2);
    for (size_t i = 0; i < n; ++i) {
        const float   v = std::max(-1.0f, std::min(1.0f, pcm[i]));
        const int16_t s = static_cast<int16_t>(v * 32767.0f);
        const uint16_t u = static_cast<uint16_t>(s);
        out.push_back(static_cast<char>(u & 0xff));
        out.push_back(static_cast<char>((u >> 8) & 0xff));
    }
}

namespace detail {
inline void put_u32le(std::string & out, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
    }
}
inline void put_u16le(std::string & out, uint16_t v) {
    out.push_back(static_cast<char>(v & 0xff));
    out.push_back(static_cast<char>((v >> 8) & 0xff));
}
}  // namespace detail

// 44-byte WAV header for mono 16-bit PCM whose length is not known yet. The
// RIFF and data chunk sizes are 0xFFFFFFFF, the conventional "unknown /
// streaming" value that players and ffmpeg accept for live streams.
inline std::string streaming_wav_header(uint32_t sample_rate) {
    std::string h;
    h.reserve(44);
    h += "RIFF";
    detail::put_u32le(h, 0xFFFFFFFFu);
    h += "WAVE";
    h += "fmt ";
    detail::put_u32le(h, 16);               // fmt chunk size
    detail::put_u16le(h, 1);                // PCM
    detail::put_u16le(h, 1);                // mono
    detail::put_u32le(h, sample_rate);
    detail::put_u32le(h, sample_rate * 2);  // byte rate
    detail::put_u16le(h, 2);                // block align
    detail::put_u16le(h, 16);               // bits per sample
    h += "data";
    detail::put_u32le(h, 0xFFFFFFFFu);
    return h;
}

}  // namespace ttsfile
