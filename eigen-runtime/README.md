##### Dynamic Runtime
This runtime consists of only one file (**dynamicEigen.cpp**) and the script
used to generate it (**dynamicGen.py**). There is no header for this part of the
runtime because it is not meant to be included by anything. Calls to this
portion of the library will be inserted into other binaries by the opt pass,
allowing them to be resolved at runtime to the functions in the library.

It is important to note that the functions here use Eigen _dynamically_ (i.e.
they do not need to know the matrix sizes at compile time); this means that we
can call these functions _statically_. In short: **the dynamic runtime uses
static calls**.

#### Files in this directory
 * **dynamicGen.py**:
    Used to generate dynamicEigen.cpp. 
 * **dynamicEigen.cpp**:
    Contains templates of Eigen GEMM kernels that use the dynamic sizing option
    of Eigen's matrix templates as well as specialisations to numerical types
    that are available to external modules. These can be called statically at
    runtime by passing the matrix dimensions as part of the function signature.
