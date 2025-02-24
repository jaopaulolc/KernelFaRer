#include "Eigen/Dense"

// Convenience naming.
using namespace Eigen;
typedef Stride<Dynamic, Dynamic> DynStride;

/// Eigen dynamic sized GEMM with alpha and beta for template type.
template <typename T>
inline void gemmAB(T *A, T *B, T *C, T a, T b, StorageOptions layoutA,
                   StorageOptions layoutB, StorageOptions layoutC, uint64_t m64,
                   uint64_t n64, uint64_t k64, uint64_t lda64, uint64_t ldb64,
                   uint64_t ldc64) {
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
  eC = (a * eA * eB) + (b * eC);
}

/// Eigen dynamic sized GEMM with alpha only for template type.
template <typename T>
inline void gemmA(T *A, T *B, T *C, T a, StorageOptions layoutA,
                  StorageOptions layoutB, StorageOptions layoutC, uint64_t m64,
                  uint64_t n64, uint64_t k64, uint64_t lda64, uint64_t ldb64,
                  uint64_t ldc64) {
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
  eC = (a * eA * eB) + eC;
}

/// Eigen dynamic sized GEMM with beta only for template type.
template <typename T>
inline void gemmB(T *A, T *B, T *C, T b, StorageOptions layoutA,
                  StorageOptions layoutB, StorageOptions layoutC, uint64_t m64,
                  uint64_t n64, uint64_t k64, uint64_t lda64, uint64_t ldb64,
                  uint64_t ldc64) {
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
  eC = (eA * eB) + (b * eC);
}

/// Eigen dynamic sized GEMM with neither alpha nor beta for template type.
template <typename T>
inline void gemm(T *A, T *B, T *C, StorageOptions layoutA,
                 StorageOptions layoutB, StorageOptions layoutC, uint64_t m64,
                 uint64_t n64, uint64_t k64, uint64_t lda64, uint64_t ldb64,
                 uint64_t ldc64) {
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
  eC = (eA * eB) + eC;
}

extern "C" {

// Specialisations for int8_t.
/// Eigen dynamic sized GEMM with alpha and beta for int8_t.
void _gemmABInt8(int8_t *A, int8_t *B, int8_t *C, int8_t a, int8_t b,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<int8_t>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                 ldc);
}
/// Eigen dynamic sized GEMM with alpha only for int8_t.
void _gemmAInt8(int8_t *A, int8_t *B, int8_t *C, int8_t a,
                StorageOptions layoutA, StorageOptions layoutB,
                StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<int8_t>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with beta only for int8_t.
void _gemmBInt8(int8_t *A, int8_t *B, int8_t *C, int8_t b,
                StorageOptions layoutA, StorageOptions layoutB,
                StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<int8_t>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for int8_t.
void _gemmInt8(int8_t *A, int8_t *B, int8_t *C, StorageOptions layoutA,
               StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
               uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
               uint64_t ldc) {
  gemm<int8_t>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for int16_t.
/// Eigen dynamic sized GEMM with alpha and beta for int16_t.
void _gemmABInt16(int16_t *A, int16_t *B, int16_t *C, int16_t a, int16_t b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<int16_t>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with alpha only for int16_t.
void _gemmAInt16(int16_t *A, int16_t *B, int16_t *C, int16_t a,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<int16_t>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with beta only for int16_t.
void _gemmBInt16(int16_t *A, int16_t *B, int16_t *C, int16_t b,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<int16_t>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for int16_t.
void _gemmInt16(int16_t *A, int16_t *B, int16_t *C, StorageOptions layoutA,
                StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                uint64_t ldc) {
  gemm<int16_t>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for int32_t.
/// Eigen dynamic sized GEMM with alpha and beta for int32_t.
void _gemmABInt32(int32_t *A, int32_t *B, int32_t *C, int32_t a, int32_t b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<int32_t>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with alpha only for int32_t.
void _gemmAInt32(int32_t *A, int32_t *B, int32_t *C, int32_t a,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<int32_t>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with beta only for int32_t.
void _gemmBInt32(int32_t *A, int32_t *B, int32_t *C, int32_t b,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<int32_t>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for int32_t.
void _gemmInt32(int32_t *A, int32_t *B, int32_t *C, StorageOptions layoutA,
                StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                uint64_t ldc) {
  gemm<int32_t>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for int64_t.
/// Eigen dynamic sized GEMM with alpha and beta for int64_t.
void _gemmABInt64(int64_t *A, int64_t *B, int64_t *C, int64_t a, int64_t b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<int64_t>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with alpha only for int64_t.
void _gemmAInt64(int64_t *A, int64_t *B, int64_t *C, int64_t a,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<int64_t>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with beta only for int64_t.
void _gemmBInt64(int64_t *A, int64_t *B, int64_t *C, int64_t b,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<int64_t>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for int64_t.
void _gemmInt64(int64_t *A, int64_t *B, int64_t *C, StorageOptions layoutA,
                StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                uint64_t ldc) {
  gemm<int64_t>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for uint8_t.
/// Eigen dynamic sized GEMM with alpha and beta for uint8_t.
void _gemmABUint8(uint8_t *A, uint8_t *B, uint8_t *C, uint8_t a, uint8_t b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<uint8_t>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with alpha only for uint8_t.
void _gemmAUint8(uint8_t *A, uint8_t *B, uint8_t *C, uint8_t a,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<uint8_t>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with beta only for uint8_t.
void _gemmBUint8(uint8_t *A, uint8_t *B, uint8_t *C, uint8_t b,
                 StorageOptions layoutA, StorageOptions layoutB,
                 StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                 uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<uint8_t>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for uint8_t.
void _gemmUint8(uint8_t *A, uint8_t *B, uint8_t *C, StorageOptions layoutA,
                StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                uint64_t ldc) {
  gemm<uint8_t>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for uint16_t.
/// Eigen dynamic sized GEMM with alpha and beta for uint16_t.
void _gemmABUint16(uint16_t *A, uint16_t *B, uint16_t *C, uint16_t a,
                   uint16_t b, StorageOptions layoutA, StorageOptions layoutB,
                   StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                   uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<uint16_t>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                   ldc);
}
/// Eigen dynamic sized GEMM with alpha only for uint16_t.
void _gemmAUint16(uint16_t *A, uint16_t *B, uint16_t *C, uint16_t a,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<uint16_t>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with beta only for uint16_t.
void _gemmBUint16(uint16_t *A, uint16_t *B, uint16_t *C, uint16_t b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<uint16_t>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for uint16_t.
void _gemmUint16(uint16_t *A, uint16_t *B, uint16_t *C, StorageOptions layoutA,
                 StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                 uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                 uint64_t ldc) {
  gemm<uint16_t>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for uint32_t.
/// Eigen dynamic sized GEMM with alpha and beta for uint32_t.
void _gemmABUint32(uint32_t *A, uint32_t *B, uint32_t *C, uint32_t a,
                   uint32_t b, StorageOptions layoutA, StorageOptions layoutB,
                   StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                   uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<uint32_t>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                   ldc);
}
/// Eigen dynamic sized GEMM with alpha only for uint32_t.
void _gemmAUint32(uint32_t *A, uint32_t *B, uint32_t *C, uint32_t a,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<uint32_t>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with beta only for uint32_t.
void _gemmBUint32(uint32_t *A, uint32_t *B, uint32_t *C, uint32_t b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<uint32_t>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for uint32_t.
void _gemmUint32(uint32_t *A, uint32_t *B, uint32_t *C, StorageOptions layoutA,
                 StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                 uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                 uint64_t ldc) {
  gemm<uint32_t>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for uint64_t.
/// Eigen dynamic sized GEMM with alpha and beta for uint64_t.
void _gemmABUint64(uint64_t *A, uint64_t *B, uint64_t *C, uint64_t a,
                   uint64_t b, StorageOptions layoutA, StorageOptions layoutB,
                   StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                   uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<uint64_t>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                   ldc);
}
/// Eigen dynamic sized GEMM with alpha only for uint64_t.
void _gemmAUint64(uint64_t *A, uint64_t *B, uint64_t *C, uint64_t a,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<uint64_t>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with beta only for uint64_t.
void _gemmBUint64(uint64_t *A, uint64_t *B, uint64_t *C, uint64_t b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<uint64_t>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                  ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for uint64_t.
void _gemmUint64(uint64_t *A, uint64_t *B, uint64_t *C, StorageOptions layoutA,
                 StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                 uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                 uint64_t ldc) {
  gemm<uint64_t>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for float.
/// Eigen dynamic sized GEMM with alpha and beta for float.
void _gemmABFloat(float *A, float *B, float *C, float a, float b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<float>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                ldc);
}
/// Eigen dynamic sized GEMM with alpha only for float.
void _gemmAFloat(float *A, float *B, float *C, float a, StorageOptions layoutA,
                 StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                 uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                 uint64_t ldc) {
  gemmA<float>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with beta only for float.
void _gemmBFloat(float *A, float *B, float *C, float b, StorageOptions layoutA,
                 StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                 uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                 uint64_t ldc) {
  gemmB<float>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for float.
void _gemmFloat(float *A, float *B, float *C, StorageOptions layoutA,
                StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                uint64_t ldc) {
  gemm<float>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

// Specialisations for double.
/// Eigen dynamic sized GEMM with alpha and beta for double.
void _gemmABDouble(double *A, double *B, double *C, double a, double b,
                   StorageOptions layoutA, StorageOptions layoutB,
                   StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                   uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmAB<double>(A, B, C, a, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb,
                 ldc);
}
/// Eigen dynamic sized GEMM with alpha only for double.
void _gemmADouble(double *A, double *B, double *C, double a,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmA<double>(A, B, C, a, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with beta only for double.
void _gemmBDouble(double *A, double *B, double *C, double b,
                  StorageOptions layoutA, StorageOptions layoutB,
                  StorageOptions layoutC, uint64_t m, uint64_t n, uint64_t k,
                  uint64_t lda, uint64_t ldb, uint64_t ldc) {
  gemmB<double>(A, B, C, b, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}
/// Eigen dynamic sized GEMM with neither alpha nor beta for double.
void _gemmDouble(double *A, double *B, double *C, StorageOptions layoutA,
                 StorageOptions layoutB, StorageOptions layoutC, uint64_t m,
                 uint64_t n, uint64_t k, uint64_t lda, uint64_t ldb,
                 uint64_t ldc) {
  gemm<double>(A, B, C, layoutA, layoutB, layoutC, m, n, k, lda, ldb, ldc);
}

} // End extern C.
