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
// Language names and codes for the speech tutor. Header-only and model-free;
// see languages_test.cpp.
//

#pragma once

#include <nlohmann/json.hpp>

#include <cctype>
#include <string>
#include <utility>

namespace ttsfile {

inline std::string tutor_language_name(const std::string & code) {
    static const std::pair<const char *, const char *> names[] = {
        {"zh", "Chinese"}, {"en", "English"}, {"de", "German"}, {"it", "Italian"},
        {"pt", "Portuguese"}, {"es", "Spanish"}, {"ja", "Japanese"}, {"ko", "Korean"},
        {"fr", "French"}, {"ru", "Russian"},
    };
    for (const auto & item : names) if (code == item.first) return item.second;
    return code;
}

inline std::string tts_language_from_asr(const nlohmann::json & transcript) {
    if (!transcript.contains("language") || !transcript["language"].is_string()) return "";
    std::string name = transcript["language"].get<std::string>();
    for (char & c : name) c = (char) std::tolower((unsigned char) c);
    // Qwen3-TTS currently has language tokens for these ten languages.
    // Whisper returns their full names in verbose_json responses.
    static const std::pair<const char *, const char *> supported[] = {
        {"chinese", "zh"}, {"english", "en"}, {"german", "de"},
        {"italian", "it"}, {"portuguese", "pt"}, {"spanish", "es"},
        {"japanese", "ja"}, {"korean", "ko"}, {"french", "fr"}, {"russian", "ru"},
    };
    for (const auto & entry : supported) {
        if (name == entry.first) return entry.second;
    }
    return "";
}

}  // namespace ttsfile
