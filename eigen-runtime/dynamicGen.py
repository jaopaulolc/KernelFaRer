'''
This generator generates C++ code that exposes Eigen's dynamic sized GEMM
kernels as part of our runtime. Each function simply inlines a template expanded
for a specific type.

The generated code does NOT produce new code. The generated code is exposed
statically to calling code.
'''
import sys

# The types to specialise to.
int_sizes = [8, 16, 32, 64]
types = [('Int{}'.format(i), 'int{}_t'.format(i)) for i in int_sizes] + \
        [('Uint{}'.format(i), 'uint{}_t'.format(i)) for i in int_sizes] + \
        [('Float', 'float'), ('Double', 'double')]

# Extern lines.
beginExtern = 'extern "C" {\n\n'
endExtern = '} // End extern C.\n'

# An external function section header.
sectionHeader = \
'''// Specialisations for {}.
'''

# The header for the file.
header = \
'''#include "Eigen/Dense"

// Convenience naming.
using namespace Eigen;
typedef Stride<Dynamic, Dynamic> DynStride;
'''

# Values to fill templates.
gemmModes = ['AB', 'A', 'B', '']
gemmSigs = [
  ', {0} a, {0} b,',
  ', {0} a,',
  ', {0} b,',
  ','
]
callSigs = [
  ' a, b,',
  ' a,',
  ' b,',
  '',
]
gemmEqs = [
  '(a * eA * eB) + (b * eC)',
  '(a * eA * eB) + eC',
  '(eA * eB) + (b * eC)',
  '(eA * eB) + eC'
]
gemmModeDescs = [
  'alpha and beta',
  'alpha only',
  'beta only',
  'neither alpha nor beta'
]

# The template for the templated dynamic size Eigen functions.
kernelTemplate = \
'''/// Eigen dynamic sized GEMM with {3} for template type.
template <typename T>
inline void gemm{0}(T *A, T *B, T *C{1} StorageOptions layoutA,
                   StorageOptions layoutB, StorageOptions layoutC,
                   uint64_t m64, uint64_t n64, uint64_t k64, uint64_t lda64,
                   uint64_t ldb64, uint64_t ldc64) {{
  // Casting to match Eigen's ptrdiff_t declaration.
  auto m = static_cast<Index>(m64);
  auto n = static_cast<Index>(n64);
  auto k = static_cast<Index>(k64);
  auto lda = static_cast<Index>(lda64);
  auto ldb = static_cast<Index>(ldb64);
  auto ldc = static_cast<Index>(ldc64);

  // Column major booleans for simplifying future expressions.
  bool aColMajor = layoutA == ColMajor;
  bool bColMajor = layoutB == ColMajor;
  bool cColMajor = layoutC == ColMajor;

  // Compute the internal strides. This switches between column and row major
  // data ordering "dynamically".
  DynStride sA(aColMajor ? lda : 1, aColMajor ? 1 : lda);
  DynStride sB(bColMajor ? ldb : 1, bColMajor ? 1 : ldb);
  DynStride sC(cColMajor ? ldc : 1, cColMajor ? 1 : ldc);

  // Map the arrays into matrices.
  Map<Matrix<T, Dynamic, Dynamic, ColMajor>, 0, DynStride> eA(A, m, k, sA);
  Map<Matrix<T, Dynamic, Dynamic, ColMajor>, 0, DynStride> eB(B, k, n, sB);
  Map<Matrix<T, Dynamic, Dynamic, ColMajor>, 0, DynStride> eC(C, m, n, sC);

  // Compute GEMM.
  eC = {2};
}}
'''

# The template for the external facing dynamic Eigen function.s
externalTemplate = \
'''/// Eigen dynamic sized GEMM with {5} for {4}.
void _gemm{0}{3}({4} *A, {4} *B, {4} *C{1}
    StorageOptions layoutA, StorageOptions layoutB, StorageOptions layoutC,
    uint64_t m, uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
    uint64_t ldc) {{
  gemm{0}<{4}>(A, B, C,{2} layoutA, layoutB, layoutC, m, n, k, lda, ldb,
    ldc);
}}
'''

# The file generation function.
def genDynamicFile(filePath):
  with open(filePath, 'w') as genFile:
    # Write unchanging header.
    genFile.write(header)
    genFile.write('\n')

    # Write the templates to be used by the external facing functions.
    gemmInfos = zip(gemmModes, gemmSigs, gemmEqs, gemmModeDescs)
    for mode, sigF, eq, desc in gemmInfos:
      sig = sigF.format('T')
      genFile.write(kernelTemplate.format(mode, sig, eq, desc))
      genFile.write('\n')

    # Write the external facing functions.
    genFile.write(beginExtern)
    for (typeTitle, typeName) in types:
      genFile.write(sectionHeader.format(typeName))
      gemmInfos = zip(gemmModes, gemmSigs, callSigs, gemmModeDescs)
      for mode, sigF, callSig, desc in gemmInfos:
        sig = sigF.format(typeName)
        fn = externalTemplate.format(mode, sig, callSig, typeTitle, typeName,
                                     desc)
        genFile.write(fn)
      genFile.write('\n')
    genFile.write(endExtern)

if __name__ == '__main__':
  assert len(sys.argv) >= 2, 'Requires file argument.'
  genDynamicFile(sys.argv[1])
