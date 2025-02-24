TARGETS = noPass BLAS EIGEN BLIS MKL ESSL POLLY

.PHONY: $(TARGETS)

all: $(TARGETS)

noPass: tests.cpp
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP0
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP1
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP2
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP3
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP4

BLAS: tests.cpp
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blas" KERNEL=BLAS
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blas" KERNEL=STEP0
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blas" KERNEL=STEP1
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blas" KERNEL=STEP2
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blas" KERNEL=STEP3
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blas" KERNEL=STEP4

EIGEN: tests.cpp
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="eigen" KERNEL=EIGEN
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="eigen" KERNEL=STEP0
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="eigen" KERNEL=STEP1
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="eigen" KERNEL=STEP2
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="eigen" KERNEL=STEP3
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="eigen" KERNEL=STEP4

BLIS: tests.cpp
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blis" KERNEL=BLIS
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blis" KERNEL=STEP0
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blis" KERNEL=STEP1
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blis" KERNEL=STEP2
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blis" KERNEL=STEP3
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="blis" KERNEL=STEP4

MKL: tests.cpp
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="mkl" KERNEL=MKL
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="mkl" KERNEL=STEP0
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="mkl" KERNEL=STEP1
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="mkl" KERNEL=STEP2
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="mkl" KERNEL=STEP3
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="mkl" KERNEL=STEP4

ESSL: tests.cpp
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="essl" KERNEL=ESSL
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="essl" KERNEL=STEP0
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="essl" KERNEL=STEP1
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="essl" KERNEL=STEP2
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="essl" KERNEL=STEP3
	$(MAKE) -f Makefile.common BUILD=$@ GEMMMODE="essl" KERNEL=STEP4

POLLY: tests.cpp
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP0
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP1
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP2
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP3
	$(MAKE) -f Makefile.common BUILD=$@ KERNEL=STEP4

clean:
	$(RM) *.ll *.s
	$(MAKE) -f Makefile.common KERNEL=BLAS  clean
	$(MAKE) -f Makefile.common KERNEL=BLIS  clean
	$(MAKE) -f Makefile.common KERNEL=EIGEN clean
	$(MAKE) -f Makefile.common KERNEL=MKL   clean
	$(MAKE) -f Makefile.common KERNEL=ESSL  clean
	$(MAKE) -f Makefile.common KERNEL=STEP0 clean
	$(MAKE) -f Makefile.common KERNEL=STEP1 clean
	$(MAKE) -f Makefile.common KERNEL=STEP2 clean
	$(MAKE) -f Makefile.common KERNEL=STEP3 clean
	$(MAKE) -f Makefile.common KERNEL=STEP4 clean
