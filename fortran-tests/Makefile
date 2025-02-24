FLANG ?= flang

FFLAGS = -Wall -O3 -g \
				 -mllvm --enable-kernel-replacer \
				 -mllvm --gemmfarer-replacement-mode=cblas-interface \
				 -mllvm --debug-only=gemm-replacer-pass

SRCS = $(wildcard *.f90)
LLS = $(SRCS:.f90=.ll)

all: $(LLS)

%.ll: %.f90
	$(FLANG) $(FFLAGS) $^ -S -emit-llvm -o $@

clean:
	$(RM) *.ll
