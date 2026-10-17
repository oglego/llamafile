// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Bounded multi-producer / single-consumer job queue for ttsfile's server.
// HTTP worker threads push(); the thread that owns the model pop()s and runs
// the job. Header-only and model-free so it can be unit tested; see
// job_queue_test.cpp.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>

namespace ttsfile {

template <typename Job>
class job_queue {
  public:
    explicit job_queue(size_t max_pending) : max_pending_(max_pending) {}

    // Returns false (and does not enqueue) if the queue is closed or full.
    bool push(std::shared_ptr<Job> job) {
        std::lock_guard<std::mutex> lock(mu_);
        if (closed_ || pending_.size() >= max_pending_) {
            return false;
        }
        pending_.push_back(std::move(job));
        cv_.notify_one();
        return true;
    }

    // After close(), push() fails; pop() keeps returning already-queued jobs
    // and then returns null, so nothing that was accepted is dropped.
    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        cv_.notify_all();
    }

    // Blocks for the next job. Returns null once closed AND drained.
    std::shared_ptr<Job> pop() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return closed_ || !pending_.empty(); });
        if (pending_.empty()) {
            return nullptr;
        }
        auto job = std::move(pending_.front());
        pending_.pop_front();
        return job;
    }

  private:
    const size_t                    max_pending_;
    std::mutex                      mu_;
    std::condition_variable         cv_;
    std::deque<std::shared_ptr<Job>> pending_;
    bool                            closed_ = false;
};

}  // namespace ttsfile
