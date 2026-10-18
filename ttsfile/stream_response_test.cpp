// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Tests ttsfile/stream_response.h against a real httplib server on a loopback
// port, with a fake producer standing in for the synthesis thread. No model.
//
//   g++ -std=c++17 -pthread -iquote ttsfile -isystem llama.cpp/vendor ttsfile/stream_response_test.cpp -o /tmp/srt2 && /tmp/srt2

#include "stream_response.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
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

using namespace std::chrono_literals;
using clk = std::chrono::steady_clock;

static double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

// Scenario knobs, set per request through the query string.
struct scenario {
    int  n_chunks  = 5;
    int  chunk_len = 960;
    int  gap_ms    = 100;
    bool fail_mid  = false;  // fail() after two chunks instead of finish()
    bool wav       = true;
};

int main() {
    httplib::Server svr;
    std::atomic<int>  pushed{0};
    std::atomic<bool> producer_saw_cancel{false};
    std::vector<std::thread> producers;
    std::mutex producers_mu;

    svr.Get("/s", [&](const httplib::Request & req, httplib::Response & res) {
        scenario sc;
        if (req.has_param("n"))    sc.n_chunks  = std::stoi(req.get_param_value("n"));
        if (req.has_param("gap"))  sc.gap_ms    = std::stoi(req.get_param_value("gap"));
        if (req.has_param("fail")) sc.fail_mid  = true;
        if (req.has_param("pcm"))  sc.wav       = false;

        auto ch = std::make_shared<ttsfile::audio_channel>();
        {
            std::lock_guard<std::mutex> lock(producers_mu);
            producers.emplace_back([ch, sc, &pushed, &producer_saw_cancel] {
                for (int i = 0; i < sc.n_chunks; ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(sc.gap_ms));
                    // chunk i is filled with the byte value 'A' + i
                    if (!ch->push(std::string(sc.chunk_len, (char) ('A' + i % 26)))) {
                        producer_saw_cancel = true;
                        return;
                    }
                    ++pushed;
                    if (sc.fail_mid && i == 1) {
                        ch->fail("synthetic failure");
                        return;
                    }
                }
                ch->finish();
            });
        }
        ch->wait_ready();
        ttsfile::stream_audio_response(res, ch, sc.wav, 24000);
    });

    const int port = svr.bind_to_any_port("127.0.0.1");
    CHECK(port > 0);
    std::thread server([&] { svr.listen_after_bind(); });
    svr.wait_until_ready();

    // ---- 1. progressive delivery, correct bytes, correct framing -----------
    {
        httplib::Client cli("127.0.0.1", port);
        cli.set_read_timeout(10, 0);
        std::string body;
        std::vector<double> arrivals;
        std::string ctype, rate_hdr;
        const auto t0 = clk::now();
        auto r = cli.Get("/s?n=5&gap=150",
                         [&](const httplib::Response & resp) {
                             ctype    = resp.get_header_value("Content-Type");
                             rate_hdr = resp.get_header_value("X-Sample-Rate");
                             return true;
                         },
                         [&](const char * data, size_t len) {
                             arrivals.push_back(secs_since(t0));
                             body.append(data, len);
                             return true;
                         });
        const double total = secs_since(t0);
        CHECK(r && r->status == 200);
        CHECK(ctype == "audio/wav");
        CHECK(rate_hdr == "24000");
        CHECK(body.size() == 44 + 5 * 960);
        CHECK(body.compare(0, 4, "RIFF") == 0 && body.compare(36, 4, "data") == 0);
        for (int i = 0; i < 5; ++i) {
            CHECK(body.compare(44 + i * 960, 960, std::string(960, (char) ('A' + i))) == 0);
        }
        // progressive: the first bytes show up long before the stream ends
        CHECK(!arrivals.empty());
        CHECK(total > 0.6);                    // 5 chunks x 150ms of producing
        CHECK(arrivals.front() < total * 0.6); // first data well before the end
        std::printf("  [1] first data at %.2fs, finished at %.2fs, %zu receive calls\n",
                    arrivals.empty() ? -1.0 : arrivals.front(), total, arrivals.size());
    }

    // ---- 2. pcm: no header, same bytes --------------------------------------
    {
        httplib::Client cli("127.0.0.1", port);
        auto r = cli.Get("/s?n=3&gap=20&pcm=1");
        CHECK(r && r->status == 200);
        CHECK(r->get_header_value("Content-Type") == "audio/pcm");
        CHECK(r->body.size() == 3 * 960);
        CHECK(r->body.compare(0, 960, std::string(960, 'A')) == 0);
        CHECK(r->body.compare(2 * 960, 960, std::string(960, 'C')) == 0);
    }

    // ---- 3. client disconnects mid-stream: the producer is told to stop -----
    {
        pushed = 0;
        producer_saw_cancel = false;
        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(10, 0);
            int received_calls = 0;
            // abort after the first piece of audio, as a listener who hangs up would
            cli.Get("/s?n=400&gap=20",
                    [](const httplib::Response &) { return true; },
                    [&](const char *, size_t) { return ++received_calls < 3; });
        }  // client destroyed -> socket closed
        // give the server a moment to notice on its next write
        for (int i = 0; i < 100 && !producer_saw_cancel; ++i) {
            std::this_thread::sleep_for(50ms);
        }
        CHECK(producer_saw_cancel);
        CHECK(pushed < 400);
        std::printf("  [3] producer stopped after %d of 400 chunks\n", pushed.load());
    }

    // ---- 4. failure mid-stream: connection is aborted, not ended cleanly ----
    {
        httplib::Client cli("127.0.0.1", port);
        cli.set_read_timeout(10, 0);
        size_t got = 0;
        auto r = cli.Get("/s?n=6&gap=30&fail=1",
                         [](const httplib::Response &) { return true; },
                         [&](const char * /*data*/, size_t len) { got += len; return true; });
        CHECK(!r);                              // transfer did not complete cleanly
        CHECK(got >= 44);                       // header (and some audio) did arrive
        std::printf("  [4] received %zu bytes before the abort\n", got);
    }

    svr.stop();
    server.join();
    {
        std::lock_guard<std::mutex> lock(producers_mu);
        for (auto & t : producers) {
            if (t.joinable()) t.join();
        }
    }

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    puts("stream_response_test: all checks passed");
    return 0;
}
