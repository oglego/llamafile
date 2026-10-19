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

#include "tutor.h"

#include "languages.h"
#include "speech_request.h"

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <deque>
#include <string>
#include <vector>

namespace ttsfile {

bool generate_tutor_reply(tutor_engine & e, const std::string & text, const std::string & lang,
                          const std::string & session_id, std::string & reply, std::string * err,
                          int * n_tokens) {
    if (!e.init || !e.init->model() || !e.init->context() || !e.init->sampler(0)) {
        *err = "tutor chat model is not initialized";
        return false;
    }
    llama_model * model = e.init->model();
    llama_context * ctx = e.init->context();
    common_sampler * smpl = e.init->sampler(0);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string language = lang.empty() ? "the language used by the learner" : tutor_language_name(lang);
    std::string system = "You are a friendly language-practice partner. Reply in " + language +
        ". Respond naturally to what the learner said, then include at most one brief, gentle correction "
        "if their wording has a clear mistake. Keep the entire spoken reply to one or two short sentences. "
        "Use plain text only: no markdown, lists, labels, or explanations about being an AI.";

    std::deque<tutor_turn> & history = e.sessions[session_id];
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    roles.reserve(2 + history.size() * 2);
    contents.reserve(2 + history.size() * 2);
    roles.push_back("system");
    contents.push_back(std::move(system));
    for (const tutor_turn & turn : history) {
        roles.push_back("user");
        contents.push_back(turn.user);
        roles.push_back("assistant");
        contents.push_back(turn.assistant);
    }
    roles.push_back("user");
    contents.push_back(text);

    std::vector<llama_chat_message> messages;
    messages.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    }
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        *err = "tutor model has no chat template";
        return false;
    }
    std::vector<char> formatted(32768);
    int32_t formatted_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                       formatted.data(), formatted.size());
    if (formatted_len < 0) {
        *err = "could not format tutor chat prompt";
        return false;
    }
    if ((size_t) formatted_len > formatted.size()) {
        formatted.resize((size_t) formatted_len + 1);
        formatted_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                  formatted.data(), formatted.size());
        if (formatted_len < 0) {
            *err = "could not format tutor chat prompt";
            return false;
        }
    }
    std::string prompt(formatted.data(), (size_t) formatted_len);
    const int32_t n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, false, true);
    if (n_prompt <= 0 || n_prompt > llama_n_ctx(ctx)) {
        *err = "tutor prompt is empty or exceeds the chat model context";
        return false;
    }
    std::vector<llama_token> tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), false, true) < 0) {
        *err = "could not tokenize tutor prompt";
        return false;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    common_sampler_reset(smpl);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token token = LLAMA_TOKEN_NULL;
    reply.clear();
    int n_gen = 0;
    for (int i = 0; i < 160; ++i) {
        if (llama_decode(ctx, batch) != 0) {
            *err = "tutor chat model decode failed";
            return false;
        }
        token = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, token, true);
        if (llama_vocab_is_eog(vocab, token)) break;
        ++n_gen;
        char piece[4096];
        const int n_piece = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (n_piece < 0) {
            *err = "could not decode tutor response token";
            return false;
        }
        reply.append(piece, (size_t) n_piece);
        batch = llama_batch_get_one(&token, 1);
    }
    if (n_tokens) *n_tokens = n_gen;
    reply = ttsfile::detail::trim(reply);
    if (reply.empty()) {
        *err = "tutor chat model returned an empty reply";
        return false;
    }
    if (!session_id.empty()) {
        history.push_back({text, reply});
        while (history.size() > 3) history.pop_front();
        if (e.sessions.size() > 128) e.sessions.erase(e.sessions.begin());
    }
    return true;
}

}  // namespace ttsfile
