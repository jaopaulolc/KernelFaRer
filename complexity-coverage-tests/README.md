# Complexity Coverage Tests

The tests here measure the execution time of different implementations of GeMM (General Matrix-Matrix Multiply).

The source file `tests.cpp` contains 5 GeMM implementations:

* `STEP0`: Naive triple-nested loop that computes GeMM
* `STEP1`: Same as `STEP0` but using `Transpose(A)` instead of `A` directly.
* `STEP2`: Same as `STEP1` but with Loop Interchange `(i, j, p) -> (j, p, i)`
* `STEP3`: Same as `STEP2` but with tiling on all 3 GeMM dimensions.
* `STEP4`: Same as `STEP3` but using packed versions of `A` and `B` instead of said matrices directly.


The makefile script is configure to run KernelFaRer and replace the GeMM loop nests with one of 4 BLAS libraries (OpenBLAS, BLIS, MKL, and ESSL).
KernelFaRer also provides an Eigen runtime that can be called to compute GeMM.  

For instance, to compile both the `Clang -O3` (`noPass`) and Eigen-replaced version (`EIGEN`) run:

~~~bash
complexity-coverage-tests $ make CLANGPLUS="<path to KernelFarer enabled Clang>"/clang++ noPass EIGEN 
~~~

To run the STEP0 replaced with Eigen run:

~~~bash
complexity-coverage-tests $ LD_LIBRARY_PATH="<path to KernelFaRer Eigen Runtime (libeigen-runtime.so)>" ./STEP0.EIGEN
~~~
