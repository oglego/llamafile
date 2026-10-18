// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Unit tests for ttsfile/audio_channel.h. No model required.
//
//   g++ -std=c++17 -pthread -iquote ttsfile ttsfile/audio_channel_test.cpp -o /tmp/act && /tmp/act

#include "audio_channel.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

static int g_failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

using ttsfile::audio_channel;
using namespace std::chrono_literals;

int main() {
    // order preserved, then done
    {
        audio_channel ch;
        std::string out;
        CHECK(ch.push("a"));
        CHECK(ch.push("bb"));
        ch.finish();
        CHECK(ch.pop(out) == audio_channel::result::chunk && out == "a");
        CHECK(ch.pop(out) == audio_channel::result::chunk && out == "bb");
        CHECK(ch.pop(out) == audio_channel::result::done);
        CHECK(ch.pop(out) == audio_channel::result::done);  // stays done
        CHECK(!ch.failed_before_audio());
    }

    // fail after data: data is delivered first, then the error
    {
        audio_channel ch;
        std::string out;
        ch.push("x");
        ch.fail("boom");
        CHECK(!ch.failed_before_audio());  // audio existed
        CHECK(ch.pop(out) == audio_channel::result::chunk && out == "x");
        CHECK(ch.pop(out) == audio_channel::result::error);
        CHECK(ch.error() == "boom");
    }

    // fail before any data is distinguishable (handler can still send a 500)
    {
        audio_channel ch;
        ch.fail("no audio");
        ch.wait_ready();  // must not block
        CHECK(ch.failed_before_audio());
        CHECK(ch.error() == "no audio");
    }

    // cancel: push fails afterwards, flag visible to the producer
    {
        audio_channel ch;
        CHECK(!ch.cancelled());
        CHECK(ch.push("a"));
        ch.cancel();
        CHECK(ch.cancelled());
        CHECK(!ch.push("b"));
    }

    // pop blocks until the producer pushes, and wakes on finish
    {
        audio_channel ch;
        std::atomic<int> chunks{0};
        std::atomic<bool> finished{false};
        std::thread consumer([&] {
            std::string out;
            for (;;) {
                auto r = ch.pop(out);
                if (r == audio_channel::result::chunk) {
                    ++chunks;
                } else {
                    finished = (r == audio_channel::result::done);
                    return;
                }
            }
        });
        std::this_thread::sleep_for(50ms);
        CHECK(chunks == 0);
        for (int i = 0; i < 5; ++i) {
            ch.push("c");
        }
        std::this_thread::sleep_for(50ms);
        ch.finish();
        consumer.join();
        CHECK(chunks == 5);
        CHECK(finished);
    }

    // wait_ready wakes on the first chunk
    {
        audio_channel ch;
        std::atomic<bool> ready{false};
        std::thread waiter([&] {
            ch.wait_ready();
            ready = true;
        });
        std::this_thread::sleep_for(50ms);
        CHECK(!ready);
        ch.push("first");
        waiter.join();
        CHECK(ready);
        CHECK(!ch.failed_before_audio());
    }

    // a consumer blocked in pop() wakes when the client cancels
    {
        audio_channel ch;
        std::atomic<bool> returned{false};
        std::thread consumer([&] {
            std::string out;
            (void) ch.pop(out);
            returned = true;
        });
        std::this_thread::sleep_for(50ms);
        CHECK(!returned);
        ch.cancel();
        consumer.join();
        CHECK(returned);
    }

    // producer/consumer stress: everything arrives, in order
    {
        audio_channel ch;
        constexpr int kN = 5000;
        std::thread producer([&] {
            for (int i = 0; i < kN; ++i) {
                ch.push(std::to_string(i));
            }
            ch.finish();
        });
        int expect = 0;
        std::string out;
        while (ch.pop(out) == audio_channel::result::chunk) {
            if (out != std::to_string(expect)) {
                ++g_failures;
                break;
            }
            ++expect;
        }
        producer.join();
        CHECK(expect == kN);
    }

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    puts("audio_channel_test: all checks passed");
    return 0;
}
