// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
// Licensed under the Apache License, Version 2.0.
//
// Unit tests for ttsfile/argv_utils.h. No model required.
//
//   g++ -std=c++17 -iquote ttsfile ttsfile/argv_utils_test.cpp -o /tmp/aut && /tmp/aut

#include "argv_utils.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                            \
        }                                                                            \
    } while (0)

// Builds a NULL-terminated argv from strings. `store` owns the buffers.
struct Args {
    std::vector<std::string> store;
    std::vector<char *>      ptrs;
    int                      argc = 0;

    explicit Args(std::vector<std::string> a) : store(std::move(a)) {
        for (auto & s : store) {
            ptrs.push_back(&s[0]);
        }
        ptrs.push_back(nullptr);
        argc = (int) store.size();
    }
    char ** argv() { return ptrs.data(); }
    std::string joined() {
        std::string out;
        for (int i = 0; i < argc; ++i) {
            out += (i ? " " : "") + std::string(ptrs[i]);
        }
        return out;
    }
    bool terminated() { return ptrs[argc] == nullptr; }
};

int main() {
    // take_flag: present in the middle
    {
        Args a({"ttsfile", "-m", "x.gguf", "--server", "-p", "hi"});
        CHECK(ttsfile::take_flag(a.argc, a.argv(), "--server"));
        CHECK(a.argc == 5 && a.joined() == "ttsfile -m x.gguf -p hi" && a.terminated());
    }
    // take_flag: last argument, and absent
    {
        Args a({"ttsfile", "-p", "hi", "--server"});
        CHECK(ttsfile::take_flag(a.argc, a.argv(), "--server"));
        CHECK(a.argc == 3 && a.joined() == "ttsfile -p hi" && a.terminated());
        CHECK(!ttsfile::take_flag(a.argc, a.argv(), "--server"));
        CHECK(a.argc == 3 && a.terminated());
    }
    // take_flag never treats argv[0] as a flag
    {
        Args a({"--server"});
        CHECK(!ttsfile::take_flag(a.argc, a.argv(), "--server"));
        CHECK(a.argc == 1);
    }
    // take_value: separate value
    {
        Args a({"ttsfile", "--port", "9000", "-p", "hi"});
        std::string v;
        bool missing = false;
        CHECK(ttsfile::take_value(a.argc, a.argv(), "--port", &v, &missing));
        CHECK(v == "9000" && !missing);
        CHECK(a.argc == 3 && a.joined() == "ttsfile -p hi" && a.terminated());
    }
    // take_value: --flag=value form, at the end
    {
        Args a({"ttsfile", "-p", "hi", "--host=0.0.0.0"});
        std::string v;
        bool missing = false;
        CHECK(ttsfile::take_value(a.argc, a.argv(), "--host", &v, &missing));
        CHECK(v == "0.0.0.0" && a.argc == 3 && a.joined() == "ttsfile -p hi" && a.terminated());
    }
    // take_value: value-taking flag as the very last argument -> missing
    {
        Args a({"ttsfile", "-p", "hi", "--port"});
        std::string v = "unchanged";
        bool missing = false;
        CHECK(!ttsfile::take_value(a.argc, a.argv(), "--port", &v, &missing));
        CHECK(missing && v == "unchanged" && a.argc == 4);
    }
    // take_value: a longer flag sharing the prefix is not matched
    {
        Args a({"ttsfile", "--port-range", "1"});
        std::string v;
        bool missing = false;
        CHECK(!ttsfile::take_value(a.argc, a.argv(), "--port", &v, &missing));
        CHECK(!missing && a.argc == 3);
    }
    // several wrapper flags removed in sequence leave upstream's args intact
    {
        Args a({"ttsfile", "--server", "-hf", "u/m", "--host", "::1", "--port=8082", "--tts-lang", "de"});
        std::string host, port;
        bool missing = false;
        CHECK(ttsfile::take_flag(a.argc, a.argv(), "--server"));
        CHECK(ttsfile::take_value(a.argc, a.argv(), "--host", &host, &missing));
        CHECK(ttsfile::take_value(a.argc, a.argv(), "--port", &port, &missing));
        CHECK(host == "::1" && port == "8082");
        CHECK(a.joined() == "ttsfile -hf u/m --tts-lang de" && a.terminated());
    }

    if (g_failures) {
        fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    puts("argv_utils_test: all checks passed");
    return 0;
}
