# C++ Unit Tests

The tests here are similar to the lit tests under [llvm/test/Transforms/GEMMFaRer](https://github.com/jaopaulolc/KernelFaRer/tree/kernelfarer/llvm/test/Transforms/GEMMFaRer) but written in C++ functions.

They test if the matcher in KernelFaRer can match different variations of functions that implement a GeMM (General Matrix-Matrix Multiply).

They can be execute as follows:

~~~bash
cplus-tests $ make CLANGPLUS=<path to clang>/clang++ PLUGIN=<path to plugin>/KernelFaRer.so 2>&1 | grep 'Found kernel is rewritable!'
~~~

The previous command line show generate an output similar to the follow:

~~~bash
gemm-0.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-1.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-10.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-11.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-12.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-13.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-14.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-15.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-2.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-3.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-4.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-5.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-6.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-7.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-8.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
gemm-9.cc:3:3: remark: Found Kernel is rewritable! [-Rpass-analysis=kernel-replacer-pass]
~~~
