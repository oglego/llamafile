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

#include "asr_process.h"

#include "log.h"

#include <cpp-httplib/httplib.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <spawn.h>
#include <string>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

namespace ttsfile {

bool start_embedded_asr(const std::string & executable, const std::string & model_path,
                        pid_t & pid, std::string & err) {
    char * argv[] = {
        const_cast<char *>(executable.c_str()),
        const_cast<char *>("-m"), const_cast<char *>(model_path.c_str()),
        const_cast<char *>("--host"), const_cast<char *>("127.0.0.1"),
        const_cast<char *>("--port"), const_cast<char *>("8082"),
        const_cast<char *>("--no-default-args"),
        nullptr,
    };
    const int rc = posix_spawn(&pid, executable.c_str(), nullptr, nullptr, argv, environ);
    if (rc != 0) {
        err = "could not start bundled whisper-server: " + std::string(strerror(rc));
        return false;
    }
    LOG_INF("starting bundled whisper-server on 127.0.0.1:%d\n", kEmbeddedAsrPort);

    httplib::Client health("127.0.0.1", kEmbeddedAsrPort);
    health.set_connection_timeout(1, 0);
    health.set_read_timeout(1, 0);
    for (int i = 0; i < 360; ++i) {
        int status = 0;
        const pid_t exited = waitpid(pid, &status, WNOHANG);
        if (exited == pid) {
            pid = -1;
            err = "bundled whisper-server exited before becoming ready";
            return false;
        }
        auto response = health.Get("/health");
        if (response && response->status == 200) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    err = "timed out waiting for bundled whisper-server to load its model";
    kill(pid, SIGTERM);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    pid = -1;
    return false;
}

void stop_embedded_asr(pid_t & pid) {
    if (pid <= 0) return;
    kill(pid, SIGTERM);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    pid = -1;
}

}  // namespace ttsfile
