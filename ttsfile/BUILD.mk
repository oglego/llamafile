#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘
#
# Copyright 2026 Mozilla.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# BUILD.mk for ttsfile
#
# ttsfile wraps llama.cpp's tools/tts (llama-tts) with llamafile features:
# GPU support, /zip/.args defaults, and embedded model/mmproj/speaker files.
#
# Like diffusionfile, the upstream CLI's main() is renamed so ttsfile's own
# main() can call it. Here that is done with -Dmain=tts_cli_main on the one
# object file, so no patch to llama.cpp is needed.

PKGS += TTSFILE

# ==============================================================================
# Objects
# ==============================================================================

TTSFILE_OBJS := \
	o/$(MODE)/ttsfile/ttsfile.o

# upstream tools/tts/tts.cpp, with main() renamed
TTSFILE_UPSTREAM_OBJ := o/$(MODE)/ttsfile/tts.upstream.o

# ==============================================================================
# Flags
# ==============================================================================
# Must match how llama.cpp/BUILD.mk builds the objects in llama.cpp.a and
# MTMD_OBJS, since tts.cpp includes common/mtmd headers whose layouts depend
# on these macros.

TTSFILE_INCLUDES := \
	-iquote . \
	-iquote ttsfile \
	-iquote llama.cpp/common \
	-iquote llama.cpp/include \
	-iquote llama.cpp/ggml/include \
	-iquote llama.cpp/tools/mtmd \
	-isystem llama.cpp/vendor

TTSFILE_CPPFLAGS := \
	$(TTSFILE_INCLUDES) \
	-DCOSMOCC=1 \
	-DGGML_MULTIPLATFORM \
	-DLLAMA_SUBPROCESS

# ==============================================================================
# Compilation rules
# ==============================================================================

o/$(MODE)/ttsfile/ttsfile.o: ttsfile/ttsfile.cpp ttsfile/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(TTSFILE_CPPFLAGS) -o $@ $<

$(TTSFILE_UPSTREAM_OBJ): llama.cpp/tools/tts/tts.cpp ttsfile/BUILD.mk $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) $(TTSFILE_CPPFLAGS) -Dmain=tts_cli_main -o $@ $<

# ==============================================================================
# Dependencies required by the upstream TTS CLI
# ==============================================================================

TTSFILE_LLAMAFILE_OBJS := \
	$(TOOL_LLAMAFILE_OBJS) \
	o/$(MODE)/llamafile/check_cpu.o

# TTS can use llama.cpp's Metal backend on macOS. Embed the upstream sources
# that llamafile/metal.c extracts and compiles at runtime.
TTSFILE_METAL_SOURCES := $(LLAMAFILE_METAL_SOURCES)

# ==============================================================================
# Executable
# ==============================================================================

o/$(MODE)/ttsfile/ttsfile: \
		$(TTSFILE_OBJS) \
		$(TTSFILE_UPSTREAM_OBJ) \
		$(MTMD_OBJS) \
		$(HTTPLIB_OBJS) \
		$(TTSFILE_LLAMAFILE_OBJS) \
		$(TTSFILE_METAL_SOURCES) \
		$(TINYBLAS_CPU_OBJS) \
		o/$(MODE)/llama.cpp/llama.cpp.a \
		o/$(MODE)/third_party/mbedtls/mbedtls.a
	@mkdir -p $(@D)
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

o/$(MODE)/ttsfile/ttsfile: private LDFLAGS += -fopenmp
o/$(MODE)/ttsfile/ttsfile: private LDLIBS += -lpthread -lm

# ==============================================================================
# Main target
# ==============================================================================

.PHONY: o/$(MODE)/ttsfile
o/$(MODE)/ttsfile: \
	o/$(MODE)/ttsfile/ttsfile
