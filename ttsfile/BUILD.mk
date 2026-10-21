#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# BUILD.mk for ttsfile.
#
# ttsfile is a single-file text-to-speech CLI. The entry point is
# ttsfile/main.cpp, a thin wrapper that loads default arguments from the
# executable's own zip store (/zip/.args, via cosmo_args) and then calls
# llama.cpp's TTS tool. That tool (llama.cpp/tools/tts/tts.cpp) is compiled by
# llama.cpp/BUILD.mk as $(TOOL_TTS_OBJS) with main() renamed to
# tts_cli_main(). Everything links against the cosmocc-built llama.cpp.a plus
# libmtmd ($(MTMD_OBJS)), which holds the Qwen3-TTS and Pocket TTS pipelines.
#
# Link set mirrors agentfile (same llama.cpp.a + llamafile GPU/zip support).

PKGS += TTSFILE

# Version string surfaced by `ttsfile --version`.
TTSFILE_VERSION_STRING := $(shell git describe --tags --always 2>/dev/null || echo "0.0.0-dev")

# ==============================================================================
# Sources
# ==============================================================================

TTSFILE_SRCS_CPP := \
	ttsfile/main.cpp

TTSFILE_OBJS := $(TTSFILE_SRCS_CPP:%.cpp=o/$(MODE)/%.cpp.o)

# ==============================================================================
# Compilation Rules
# ==============================================================================

# COSMOCC enables the cosmo_args("/zip/.args") path, as in llamafile/BUILD.mk.
TTSFILE_CPPFLAGS := \
	-DCOSMOCC=1 \
	-DTTSFILE_VERSION_STRING=\"$(TTSFILE_VERSION_STRING)\"

$(TTSFILE_OBJS): o/$(MODE)/%.cpp.o: %.cpp ttsfile/BUILD.mk llamafile/llamafile.h $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(TTSFILE_CPPFLAGS) -o $@ $<

# ==============================================================================
# Executable
# ==============================================================================
# Links:
#   - ttsfile/main.cpp (entry point)
#   - $(TOOL_TTS_OBJS): upstream tts.cpp with main() renamed
#   - $(MTMD_OBJS): libmtmd (audio generation pipelines + audio decoding)
#   - llama.cpp.a (ggml, llama, common)
#   - $(TOOL_LLAMAFILE_OBJS): runtime GPU dispatch + zip filesystem
#   - $(LLAMAFILE_METAL_SOURCES): embedded Metal kernel sources (macOS)
#   - $(HTTPLIB_OBJS) + mbedtls.a: common/download.cpp (-hf) needs them

o/$(MODE)/ttsfile/ttsfile: \
		$(TTSFILE_OBJS) \
		$(TOOL_TTS_OBJS) \
		$(MTMD_OBJS) \
		$(HTTPLIB_OBJS) \
		$(TOOL_LLAMAFILE_OBJS) \
		$(LLAMAFILE_METAL_SOURCES) \
		$(TINYBLAS_CPU_OBJS) \
		o/$(MODE)/llama.cpp/llama.cpp.a \
		o/$(MODE)/third_party/mbedtls/mbedtls.a
	@mkdir -p $(@D)
	$(LINK.o) $(TTSFILE_OBJS) $(TOOL_TTS_OBJS) $(MTMD_OBJS) $(HTTPLIB_OBJS) $(TOOL_LLAMAFILE_OBJS) $(LLAMAFILE_METAL_SOURCES) $(TINYBLAS_CPU_OBJS) o/$(MODE)/llama.cpp/llama.cpp.a o/$(MODE)/third_party/mbedtls/mbedtls.a $(LOADLIBES) $(LDLIBS) -fopenmp -lpthread -o $@

# ==============================================================================
# Main Target
# ==============================================================================

.PHONY: o/$(MODE)/ttsfile
o/$(MODE)/ttsfile: o/$(MODE)/ttsfile/ttsfile
