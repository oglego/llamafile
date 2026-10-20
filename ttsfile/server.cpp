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

#include "server.h"

#include "audio_channel.h"
#include "audio_encode.h"
#include "job_queue.h"
#include "languages.h"
#include "speech_request.h"
#include "stream_response.h"
#include "timing.h"
#include "web_ui.h"

#include "log.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <future>
#include <memory>
#include <string>
#include <thread>

using namespace ttsfile;

namespace {

// ---------------------------------------------------------------------------
// HTTP server
//
// Threading model (same idea as llama-server): httplib worker threads only
// parse requests and enqueue jobs. ALL synthesis runs on the thread that
// loaded the model (main), one job at a time. llamafile's GPU backends must
// run on the main thread (see the yield_to_queue() comment in the server-queue
// patch), and an early version of this server that ran the model on httplib
// worker threads died with a bare "bus error". The root cause of that crash
// was never confirmed, so this is also the conservative design.
// ---------------------------------------------------------------------------

constexpr size_t kMaxQueuedJobs = 8;  // beyond this, requests get 503 instead of piling up

struct job_outcome {
    bool        ok = false;
    std::string err;
    tts_result  result;
    ttsfile::turn_timing timing;  // filled for speech-to-speech turns
};

struct tts_job {
    std::string                text;
    std::string                lang;
    std::string                session_id;
    bool                       tutor = false;
    double                     asr_s = 0.0;                              // measured by the handler
    std::chrono::steady_clock::time_point enqueued;                      // for queue-wait timing
    // exactly one of these is used:
    std::promise<job_outcome>                done;     // non-streaming: whole result
    std::shared_ptr<ttsfile::audio_channel>  channel;  // streaming: chunks as produced
};

// Runs one job on the main thread.
void run_job(tts_engine & e, tutor_engine * tutor, tts_job & job) {
    if (!job.channel) {
        job_outcome out;
        if (job.tutor) {
            out.timing.asr_s   = job.asr_s;
            out.timing.queue_s = ttsfile::seconds_since(job.enqueued);
            std::string response;
            const auto t_chat = std::chrono::steady_clock::now();
            const bool chat_ok = tutor && generate_tutor_reply(*tutor, job.text, job.lang, job.session_id,
                                                               response, &out.err, &out.timing.chat_tokens);
            out.timing.chat_s = ttsfile::seconds_since(t_chat);
            if (!chat_ok) {
                out.ok = false;
                if (out.err.empty()) out.err = "speech tutor requires --chat-model";
                job.done.set_value(std::move(out));
                return;
            }
            job.text = std::move(response);
        }
        out.ok = synthesize(e, job.text, job.lang, out.result, &out.err);
        if (job.tutor && out.ok) {
            out.timing.tts_s   = out.result.t_total_s;
            out.timing.audio_s = out.result.sample_rate > 0 ? (double) out.result.n_samples / out.result.sample_rate : 0.0;
        }
        job.done.set_value(std::move(out));
        return;
    }

    ttsfile::audio_channel & ch = *job.channel;
    if (ch.cancelled()) {  // the client left while this was still queued
        ch.finish();
        return;
    }

    stream_sink sink;
    sink.chunk_frames = e.stream_chunk_frames;
    sink.emit = [&ch](const float * pcm, size_t n) {
        std::string bytes;
        ttsfile::append_s16le(bytes, pcm, n);
        return ch.push(std::move(bytes));
    };

    tts_result  result;
    std::string err;
    if (synthesize(e, job.text, job.lang, result, &err, &sink)) {
        ch.finish();
    } else if (result.cancelled) {
        LOG_INF("client disconnected; synthesis stopped early\n");
        ch.finish();
    } else {
        LOG_ERR("streaming synthesis failed: %s\n", err.c_str());
        ch.fail(err);
    }
}

std::atomic<httplib::Server *> g_server{nullptr};

void on_signal(int) {
    if (httplib::Server * s = g_server.load()) {
        s->stop();
    }
}

void reply_error(httplib::Response & res, int status, const std::string & message) {
    const char * type = status >= 500 ? "server_error" : "invalid_request_error";
    nlohmann::json j = {{"error", {{"message", message}, {"type", type}}}};
    res.status = status;
    res.set_content(j.dump(), "application/json");
}

}  // namespace

namespace ttsfile {

int run_server(tts_engine & e, tutor_engine * tutor, const std::string & host, int port,
               const std::string & asr_url) {
    httplib::Server svr;
    ttsfile::job_queue<tts_job> queue(kMaxQueuedJobs);

    // Speech-to-speech accepts uploaded audio. Keep the upload bounded while
    // allowing ordinary recordings (up to 25 MiB).
    svr.set_payload_max_length(25 * 1024 * 1024);

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Cache-Control", "no-store");
        res.set_content(ttsfile::kWebUi, "text/html; charset=utf-8");
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        ttsfile::speech_request sr;
        const std::string perr = ttsfile::parse_speech_request(req.body, sr);
        if (!perr.empty()) {
            reply_error(res, 400, perr);
            return;
        }

        auto job  = std::make_shared<tts_job>();
        job->text = sr.input;
        job->lang = sr.lang.empty() ? e.params->tts_lang : sr.lang;
        const bool wav = sr.format == "wav";

        if (sr.stream) {
            auto ch      = std::make_shared<ttsfile::audio_channel>();
            job->channel = ch;
            if (!queue.push(job)) {
                reply_error(res, 503, "server is busy or shutting down; retry later");
                return;
            }

            // Wait for the first audio (or a failure) before committing to a
            // 200: after the first byte the status line can no longer change.
            ch->wait_ready();
            if (ch->failed_before_audio()) {
                LOG_ERR("synthesis failed: %s\n", ch->error().c_str());
                reply_error(res, 500, ch->error());
                return;
            }

            ttsfile::stream_audio_response(res, ch, wav, (uint32_t) e.sample_rate);
            return;
        }

        auto fut = job->done.get_future();
        if (!queue.push(job)) {
            reply_error(res, 503, "server is busy or shutting down; retry later");
            return;
        }

        const job_outcome out = fut.get();  // main thread fulfils this
        if (!out.ok) {
            LOG_ERR("synthesis failed: %s\n", out.err.c_str());
            reply_error(res, 500, out.err);
            return;
        }
        if (wav) {
            res.set_content(out.result.wav, "audio/wav");
        } else {
            // the helper's WAV is a fixed 44-byte header followed by s16le samples
            res.set_header("X-Sample-Rate", std::to_string(out.result.sample_rate));
            res.set_header("X-Sample-Format", "s16le");
            res.set_header("X-Channels", "1");
            res.set_content(out.result.wav.substr(44), "audio/pcm");
        }
    });

    // First speech-to-speech iteration: transcribe an uploaded recording
    // with a local whisper.cpp server, generate a concise tutor reply, then
    // speak it with the already-loaded TTS model.
    svr.Post("/v1/audio/speech-to-speech", [&](const httplib::Request & req, httplib::Response & res) {
        const auto t_request = std::chrono::steady_clock::now();
        if (!tutor) {
            reply_error(res, 503, "speech tutor requires --chat-model");
            return;
        }
        if (asr_url.empty()) {
            reply_error(res, 503, "speech-to-speech requires --asr-model or --asr-url");
            return;
        }
        if (!req.form.has_file("file")) {
            reply_error(res, 400, "multipart audio upload must include a file field named \"file\"");
            return;
        }
        const auto audio = req.form.get_file("file");
        if (audio.content.empty()) {
            reply_error(res, 400, "uploaded audio is empty");
            return;
        }

        httplib::Client asr(asr_url);
        asr.set_connection_timeout(5, 0);
        asr.set_read_timeout(300, 0);
        httplib::UploadFormDataItems fields = {
            {"file", audio.content, audio.filename.empty() ? "speech.wav" : audio.filename,
             audio.content_type.empty() ? "application/octet-stream" : audio.content_type},
            {"response_format", "verbose_json", "", ""},
            {"no_language_probabilities", "true", "", ""},
        };
        std::string lang;
        if (req.form.has_field("language")) {
            lang = req.form.get_field("language");
        } else if (req.has_param("language")) {
            lang = req.get_param_value("language");
        }
        if (!lang.empty()) {
            if (lang.size() != 2 || !std::isalpha((unsigned char) lang[0]) ||
                !std::isalpha((unsigned char) lang[1])) {
                reply_error(res, 400, "language must be a two-letter ISO 639-1 code such as \"en\"");
                return;
            }
            for (char & c : lang) c = (char) std::tolower((unsigned char) c);
            fields.push_back({"language", lang, "", ""});
        }
        const auto t_asr = std::chrono::steady_clock::now();
        auto asr_res = asr.Post("/inference", fields);
        const double asr_s = ttsfile::seconds_since(t_asr);
        if (!asr_res) {
            reply_error(res, 502, "could not reach the configured whisper-server");
            return;
        }
        if (asr_res->status < 200 || asr_res->status >= 300) {
            reply_error(res, 502, "whisper-server returned HTTP " + std::to_string(asr_res->status));
            return;
        }
        nlohmann::json transcript = nlohmann::json::parse(asr_res->body, nullptr, false);
        if (transcript.is_discarded() || !transcript.is_object() || !transcript.contains("text") ||
            !transcript["text"].is_string()) {
            reply_error(res, 502, "whisper-server returned an invalid JSON transcription");
            return;
        }
        const std::string spoken_text = ttsfile::detail::trim(transcript["text"].get<std::string>());
        if (spoken_text.empty()) {
            reply_error(res, 422, "no speech was recognized in the uploaded audio");
            return;
        }
        if (spoken_text.size() > 1000) {
            reply_error(res, 413, "recognized speech is too long (max 1000 bytes per turn)");
            return;
        }

        std::string session_id;
        if (req.form.has_field("session_id")) {
            session_id = req.form.get_field("session_id");
        } else if (req.has_param("session_id")) {
            session_id = req.get_param_value("session_id");
        }
        session_id = ttsfile::detail::trim(session_id);
        if (session_id.size() > 128) {
            reply_error(res, 400, "session_id must be at most 128 bytes");
            return;
        }

        auto job = std::make_shared<tts_job>();
        job->text = spoken_text;
        job->lang = lang.empty() ? tts_language_from_asr(transcript) : lang;
        if (job->lang.empty()) job->lang = e.params->tts_lang;
        job->session_id = session_id;
        job->tutor = true;
        job->asr_s = asr_s;
        job->enqueued = std::chrono::steady_clock::now();
        auto fut = job->done.get_future();
        if (!queue.push(job)) {
            reply_error(res, 503, "server is busy or shutting down; retry later");
            return;
        }
        const job_outcome out = fut.get();
        if (!out.ok) {
            LOG_ERR("speech-to-speech synthesis failed: %s\n", out.err.c_str());
            reply_error(res, 500, out.err);
            return;
        }
        ttsfile::turn_timing timing = out.timing;
        timing.total_s = ttsfile::seconds_since(t_request);
        LOG_INF("%s\n", ttsfile::timing_summary(timing).c_str());
        res.set_header("Server-Timing", ttsfile::server_timing_header(timing));
        res.set_content(out.result.wav, "audio/wav");
    });

    g_server.store(&svr);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    // HTTP runs on its own thread; this (main) thread does the synthesis.
    // listen() returns false only if it could not bind/listen; a clean stop()
    // from the signal handler makes it return true. Either way, close the
    // queue so the loop below ends.
    bool listened = false;
    std::thread http_thread([&] {
        LOG_INF("ttsfile listening on http://%s:%d (GET /, POST /v1/audio/speech, GET /health)\n",
                host.c_str(), port);
        listened = svr.listen(host, port);
        queue.close();
    });

    while (auto job = queue.pop()) {
        run_job(e, tutor, *job);
    }

    http_thread.join();
    g_server.store(nullptr);
    if (!listened) {
        LOG_ERR("failed to listen on %s:%d (port in use?)\n", host.c_str(), port);
        return 1;
    }
    return 0;
}

}  // namespace ttsfile
