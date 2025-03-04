CLANGPLUS ?= clang++

PLUGIN ?= ../build/passes/KernelFaRer.so
PASS = kernel-replacer-pass


CXXFLAGS = -Wall -g -O3 -ffp-contract=off \
					 -fplugin=$(PLUGIN) \
					 -fpass-plugin=$(PLUGIN) \
					 -mllvm --kernelfarer-replacement-mode=cblas-interface \
					 -Rpass-analysis=$(PASS) \
					 -Rpass-missed=$(PASS)

SRCS = $(wildcard *.cc)
LLS = $(SRCS:.cc=.ll)

all: $(LLS)

%.ll: %.cc
	$(CLANGPLUS) $(CPPFLAGS) $(CXXFLAGS) $^ -S -emit-llvm -o $@

clean:
	$(RM) *.ll
