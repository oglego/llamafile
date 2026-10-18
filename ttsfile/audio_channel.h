// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// One-way byte-chunk channel between ttsfile's synthesis thread (producer)
// and the HTTP worker thread streaming the response (consumer).
//
//   producer: push() chunks, then finish() or fail().
//   consumer: pop() until done/error; cancel() if the client went away.
//
// Cancellation is how a dropped connection stops an in-progress synthesis:
// once cancelled, push() returns false and the producer should stop.
// Header-only and model-free; see audio_channel_test.cpp.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <utility>

namespace ttsfile {

class audio_channel {
  public:
    enum class result { chunk, done, error };

    // ---- producer side ---------------------------------------------------

    // Returns false (chunk dropped) if the consumer has cancelled.
    bool push(std::string chunk) {
        std::lock_guard<std::mutex> lock(mu_);
        if (cancelled_) {
            return false;
        }
        ++n_pushed_;
        chunks_.push_back(std::move(chunk));
        cv_.notify_all();
        return true;
    }

    // Normal end of stream. Already-pushed chunks are still delivered.
    void finish() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        cv_.notify_all();
    }

    // Abnormal end. Already-pushed chunks are still delivered first, then
    // pop() reports an error.
    void fail(std::string message) {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        failed_ = true;
        error_  = std::move(message);
        cv_.notify_all();
    }

    bool cancelled() const {
        std::lock_guard<std::mutex> lock(mu_);
        return cancelled_;
    }

    // ---- consumer side ---------------------------------------------------

    // Blocks until a chunk is available or the stream has ended.
    result pop(std::string & out) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !chunks_.empty() || closed_ || cancelled_; });
        if (!chunks_.empty()) {
            out = std::move(chunks_.front());
            chunks_.pop_front();
            return result::chunk;
        }
        return failed_ ? result::error : result::done;
    }

    // The client is gone. Wakes everyone and makes push() fail.
    void cancel() {
        std::lock_guard<std::mutex> lock(mu_);
        cancelled_ = true;
        cv_.notify_all();
    }

    // Blocks until the first chunk exists or the stream has ended, so the HTTP
    // handler can still return a proper error status if synthesis fails before
    // any audio was produced (after the first byte the status line is gone).
    void wait_ready() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !chunks_.empty() || closed_ || cancelled_; });
    }

    // True if the stream failed and no audio was ever pushed.
    bool failed_before_audio() const {
        std::lock_guard<std::mutex> lock(mu_);
        return failed_ && n_pushed_ == 0;
    }

    std::string error() const {
        std::lock_guard<std::mutex> lock(mu_);
        return error_;
    }

  private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<std::string> chunks_;
    size_t                  n_pushed_  = 0;
    bool                    closed_    = false;
    bool                    failed_    = false;
    bool                    cancelled_ = false;
    std::string             error_;
};

}  // namespace ttsfile
