# KernelFaRer

> KernelFaRer: Replacing Native-Code Idioms with High-Performance Library Calls ([Preprint](https://www.researchgate.net/publication/350453412_KernelFaRer_Replacing_Native-Code_Idioms_with_High-Performance_Library_Calls))

by

> João P. L. de Carvalho, Braedy Kuzma, Ivan Korostelev, José Nelson Amaral, Christopher Barton, José Moreira, and Guido Araujo.

Please follow the instructions bellow to compile KernelFaRer:

~~~bash
$ git clone https://github.com/jaopaulolc/KernelFaRer.git
$ cd KernelFaRer
KernelFaRer $ cmake -B build -S . -DLLVM_DIR=$(llvm-config --cmakedir)
KernelFaRer $ cmake --build build
~~~

For examples, please see [C++ examples](cplus-tests)

## TODOs:

- Enable recognition in the presence of `llvm.fmuladd.*`
- Add the ability to run lit tests
