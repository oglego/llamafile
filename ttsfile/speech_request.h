// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
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

//
// Request parsing for ttsfile's POST /v1/audio/speech endpoint.
//
// Deliberately free of llama.cpp / httplib dependencies (only nlohmann json)
// so it can be unit tested without a model; see speech_request_test.cpp.
//
// The request shape follows OpenAI's audio/speech API where it makes sense:
//
//   { "input": "text to speak",          // required
//     "language": "de",                  // optional, ISO 639-1
//     "response_format": "wav",          // optional, only "wav" is supported
//     "model": "...", "voice": "..." }   // accepted and ignored
//
// "model" and "voice" are ignored on purpose: ttsfile serves exactly one
// model, and the voice comes from --tts-speaker-file at startup.
//

#pragma once

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstddef>
#include <string>

namespace ttsfile {

// Upper bound on "input" in bytes. Keeps one request from monopolising the
// (single, serialized) synthesis engine. Callers may pass a different limit.
constexpr size_t kMaxInputBytes = 2000;

struct speech_request {
    std::string input;  // trimmed text to synthesize
    std::string lang;   // lower-case ISO 639-1 code, empty = server default
};

namespace detail {

inline std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

}  // namespace detail

// Parses and validates a request body. Returns an empty string on success
// (and fills `out`), otherwise a human-readable error message suitable for
// a 400 response. `out` is only modified on success.
inline std::string parse_speech_request(const std::string & body,
                                        speech_request & out,
                                        size_t max_input_bytes = kMaxInputBytes) {
    nlohmann::json j = nlohmann::json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        return "request body must be a JSON object";
    }

    auto it = j.find("input");
    if (it == j.end() || !it->is_string()) {
        return "\"input\" (string) is required";
    }
    std::string input = detail::trim(it->get<std::string>());
    if (input.empty()) {
        return "\"input\" must not be empty";
    }
    if (input.size() > max_input_bytes) {
        return "\"input\" is too long (" + std::to_string(input.size()) + " bytes, max " +
               std::to_string(max_input_bytes) + ")";
    }

    it = j.find("response_format");
    if (it != j.end() && !it->is_null()) {
        if (!it->is_string() || it->get<std::string>() != "wav") {
            return "unsupported \"response_format\": only \"wav\" is supported";
        }
    }

    std::string lang;
    it = j.find("language");
    if (it != j.end() && !it->is_null()) {
        if (!it->is_string()) {
            return "\"language\" must be a string such as \"en\"";
        }
        lang = it->get<std::string>();
        if (lang.size() != 2) {
            return "\"language\" must be a two-letter ISO 639-1 code such as \"en\"";
        }
        for (char & c : lang) {
            const unsigned char u = static_cast<unsigned char>(c);
            if (!std::isalpha(u)) {
                return "\"language\" must be a two-letter ISO 639-1 code such as \"en\"";
            }
            c = static_cast<char>(std::tolower(u));
        }
    }

    out.input = std::move(input);
    out.lang  = std::move(lang);
    return "";
}

}  // namespace ttsfile
