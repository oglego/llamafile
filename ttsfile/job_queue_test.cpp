// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Unit tests for ttsfile/job_queue.h. No model required.
//
//   g++ -std=c++17 -pthread -iquote ttsfile ttsfile/job_queue_test.cpp -o /tmp/jqt && /tmp/jqt

#include "job_queue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <thread>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

struct job {
    int id;
    std::promise<int> done;
};

static std::shared_ptr<job> mk(int id) {
    auto j = std::make_shared<job>();
    j->id = id;
    return j;
}

int main() {
    using namespace std::chrono_literals;

    // FIFO order and bounded capacity
    {
        ttsfile::job_queue<job> q(2);
        CHECK(q.push(mk(1)));
        CHECK(q.push(mk(2)));
        CHECK(!q.push(mk(3)));  // full
        CHECK(q.pop()->id == 1);
        CHECK(q.push(mk(4)));   // space again
        CHECK(q.pop()->id == 2);
        CHECK(q.pop()->id == 4);
    }

    // close(): rejects new work but drains what was accepted, then returns null
    {
        ttsfile::job_queue<job> q(8);
        CHECK(q.push(mk(1)));
        CHECK(q.push(mk(2)));
        q.close();
        CHECK(!q.push(mk(3)));
        CHECK(q.pop()->id == 1);
        CHECK(q.pop()->id == 2);
        CHECK(q.pop() == nullptr);
        CHECK(q.pop() == nullptr);  // stays closed
    }

    // close() wakes a consumer that is blocked in pop()
    {
        ttsfile::job_queue<job> q(8);
        std::atomic<bool> returned{false};
        std::thread consumer([&] {
            CHECK(q.pop() == nullptr);
            returned = true;
        });
        std::this_thread::sleep_for(100ms);
        CHECK(!returned);  // really was blocked
        q.close();
        consumer.join();
        CHECK(returned);
    }

    // end to end: many producers, one consumer fulfilling promises, then a
    // clean shutdown -- the same shape as run_server()
    {
        ttsfile::job_queue<job> q(64);
        std::thread consumer([&] {
            while (auto j = q.pop()) {
                j->done.set_value(j->id * 2);
            }
        });

        constexpr int kProducers = 8, kPerProducer = 25;
        std::atomic<int> wrong{0}, accepted{0};
        std::vector<std::thread> producers;
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&, p] {
                for (int i = 0; i < kPerProducer; ++i) {
                    auto j = mk(p * 1000 + i);
                    auto f = j->done.get_future();
                    if (!q.push(j)) {
                        std::this_thread::sleep_for(1ms);  // full: retry once
                        if (!q.push(j)) {
                            continue;
                        }
                    }
                    ++accepted;
                    if (f.get() != j->id * 2) {
                        ++wrong;
                    }
                }
            });
        }
        for (auto & t : producers) {
            t.join();
        }
        q.close();
        consumer.join();
        CHECK(wrong == 0);
        CHECK(accepted > 0);
    }

    // jobs accepted before close() are still fulfilled during shutdown
    {
        ttsfile::job_queue<job> q(8);
        auto j = mk(21);
        auto f = j->done.get_future();
        CHECK(q.push(j));
        q.close();
        std::thread consumer([&] {
            while (auto x = q.pop()) {
                x->done.set_value(x->id * 2);
            }
        });
        CHECK(f.wait_for(2s) == std::future_status::ready && f.get() == 42);
        consumer.join();
    }

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    puts("job_queue_test: all checks passed");
    return 0;
}
