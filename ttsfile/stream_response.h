// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Turns an audio_channel into a chunked HTTP audio response. Lives in its own
// header so it can be tested against a real local httplib server without a
// model; see stream_response_test.cpp.

#pragma once

#include "audio_channel.h"
#include "audio_encode.h"

#include <cpp-httplib/httplib.h>

#include <cstdint>
#include <memory>
#include <string>

namespace ttsfile {

// Makes `res` a streamed response fed by `ch`: a WAV header (unknown length)
// followed by s16le chunks, or bare s16le for "pcm". Call after
// ch->wait_ready() has returned and a failure before the first audio has been
// ruled out, because once the first byte is on the wire the status is fixed.
//
// Each chunk is written on the calling HTTP worker thread as the synthesis
// thread produces it. If the client goes away the channel is cancelled, which
// makes the producer's push() fail so it stops generating. If the stream ends
// in an error the provider returns false, which aborts the connection so the
// client sees a truncated transfer rather than a clean end.
inline void stream_audio_response(httplib::Response & res,
                                  std::shared_ptr<audio_channel> ch,
                                  bool wav,
                                  uint32_t sample_rate) {
    res.set_header("Cache-Control", "no-store");
    res.set_header("X-Sample-Rate", std::to_string(sample_rate));
    res.set_header("X-Sample-Format", "s16le");
    res.set_header("X-Channels", "1");

    auto header_sent = std::make_shared<bool>(false);
    res.set_chunked_content_provider(
        wav ? "audio/wav" : "audio/pcm",
        [ch, wav, sample_rate, header_sent](size_t, httplib::DataSink & sink) -> bool {
            if (wav && !*header_sent) {
                *header_sent = true;
                const std::string h = streaming_wav_header(sample_rate);
                if (!sink.write(h.data(), h.size())) {
                    ch->cancel();
                    return false;
                }
            }
            std::string chunk;
            switch (ch->pop(chunk)) {
                case audio_channel::result::chunk:
                    if (!sink.write(chunk.data(), chunk.size())) {
                        ch->cancel();  // client went away: stop synthesizing
                        return false;
                    }
                    return true;
                case audio_channel::result::done:
                    sink.done();
                    return true;
                case audio_channel::result::error:
                    return false;
            }
            return false;
        },
        [ch](bool success) {
            if (!success) {
                ch->cancel();
            }
        });
}

}  // namespace ttsfile
