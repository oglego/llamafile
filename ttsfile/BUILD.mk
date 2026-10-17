#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for ttsfile (EXPERIMENTAL).
#
# ttsfile is text-to-speech on top of llama.cpp's libmtmd audio generation
# (Qwen3-TTS, Pocket TTS). Unlike transcribefile/whisperfile it does not need
# its own engine submodule: the model code already lives in llama.cpp.a and
# the generation pipelines in $(MTMD_OBJS), both built by llama.cpp/BUILD.mk
# for llama-server. ttsfile/main.cpp only adds the CLI/server shell, so the
# link line mirrors llama-tts / llama-bench.
#
# Opt-in: not part of the default o/$(MODE)/ target. Build with
#
#   make o//ttsfile                      # the executable
#   make o//ttsfile/argv_utils_test.runs o//ttsfile/job_queue_test.runs \
#        o//ttsfile/speech_request_test.runs
#
# (the unit tests need no model and no GPU.)

PKGS += TTSFILE

TTSFILE_VERSION_STRING := $(shell git describe --tags --always 2>/dev/null || echo "0.0.0-dev")

# ==============================================================================
# Sources
# ==============================================================================

TTSFILE_SRCS_CPP := ttsfile/main.cpp
TTSFILE_OBJS     := $(TTSFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# Rebuild triggers only. Deliberately NOT named TTSFILE_HDRS: PKGS *_HDRS are fed
# to mkdeps (build/deps.mk), which cannot resolve speech_request.h's
# <nlohmann/json.hpp> include.
TTSFILE_DEP_HDRS := \
	ttsfile/argv_utils.h \
	ttsfile/job_queue.h \
	ttsfile/speech_request.h

# Same include set as the llama.cpp tools. main.cpp pulls in httplib, so it
# must be built with the same TLS macro as httplib.cpp (the macro changes the
# httplib class layouts; see llama.cpp/BUILD.mk) or the server would
# corrupt memory at runtime.
TTSFILE_INCLUDES := \
	-iquote . \
	-iquote ttsfile \
	-iquote llama.cpp/common \
	-iquote llama.cpp/include \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/tools/mtmd \
	-isystem llama.cpp/vendor \
	-isystem third_party/mbedtls/include

TTSFILE_CPPFLAGS := \
	-DCPPHTTPLIB_MBEDTLS_SUPPORT \
	-DTTSFILE_VERSION_STRING=\"$(TTSFILE_VERSION_STRING)\"

$(TTSFILE_OBJS): o/$(MODE)/%.cpp.o: %.cpp ttsfile/BUILD.mk $(TTSFILE_DEP_HDRS) $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(TTSFILE_INCLUDES) $(TTSFILE_CPPFLAGS) -o $@ $<

# ==============================================================================
# Executable
# ==============================================================================
# Link line = llama-tts's: ttsfile objs + mtmd + httplib (download support in
# common/) + the llamafile bridge objects that src/llama-mmap.cpp and
# ggml/src/gguf.cpp call into + tinyblas + llama.cpp.a + mbedtls.

o/$(MODE)/ttsfile/ttsfile: private LDFLAGS += -fopenmp
o/$(MODE)/ttsfile/ttsfile: private LDLIBS += -lpthread
o/$(MODE)/ttsfile/ttsfile: \
		$(TTSFILE_OBJS) \
		$(MTMD_OBJS) \
		$(HTTPLIB_OBJS) \
		$(TOOL_LLAMAFILE_OBJS) \
		$$(TINYBLAS_CPU_OBJS) \
		o/$(MODE)/llama.cpp/llama.cpp.a \
		o/$(MODE)/third_party/mbedtls/mbedtls.a
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

# ==============================================================================
# Unit tests (no model required)
# ==============================================================================

TTSFILE_TEST_NAMES := argv_utils_test job_queue_test speech_request_test
TTSFILE_TEST_OBJS  := $(TTSFILE_TEST_NAMES:%=o/$(MODE)/ttsfile/%.cpp.o)

$(TTSFILE_TEST_OBJS): o/$(MODE)/ttsfile/%.cpp.o: ttsfile/%.cpp ttsfile/BUILD.mk $(TTSFILE_DEP_HDRS) $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) -iquote ttsfile -isystem llama.cpp/vendor -o $@ $<

$(TTSFILE_TEST_NAMES:%=o/$(MODE)/ttsfile/%): private LDLIBS += -lpthread
$(TTSFILE_TEST_NAMES:%=o/$(MODE)/ttsfile/%): o/$(MODE)/ttsfile/%: o/$(MODE)/ttsfile/%.cpp.o
	@mkdir -p $(@D)
	$(LINK.o) $< $(LOADLIBES) $(LDLIBS) -o $@

$(TTSFILE_TEST_NAMES:%=o/$(MODE)/ttsfile/%.runs): o/$(MODE)/ttsfile/%.runs: o/$(MODE)/ttsfile/%
	$<
	@touch $@

# ==============================================================================
# Main Target
# ==============================================================================

.PHONY: o/$(MODE)/ttsfile
o/$(MODE)/ttsfile: o/$(MODE)/ttsfile/ttsfile
