#include <cstdlib>
#include <iostream>
#include <chrono>
#include <random>
#include <unistd.h>
#include <cstring>
#include <functional>

#ifdef BLAS
#include <cblas.h>
#endif
#ifndef DISABLE_VERIFY
#include <eigen3/Eigen/Dense>
#endif
#ifdef BLIS
#include <blis.h>
#endif
#ifdef MKL
#include <mkl.h>
#endif
#ifdef ESSL
#include <essl.h>
#endif

#ifndef CACHE_SIZE_IN_KB
#define CACHE_SIZE_IN_KB (48*1024*1024)
#endif
#ifndef M_DIM
#define M_DIM (4*1024)
#endif
#ifndef K_DIM
#define K_DIM (4*1024)
#endif
#ifndef N_DIM
#define N_DIM (4*1024)
#endif

using namespace std;

#ifndef DISABLE_VERIFY
using namespace Eigen;
typedef Stride<Dynamic, Dynamic> DynStride;
#endif

#if defined(RUNSTEP0)
void step0_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  // Naive GEMM
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      for (p = 0; p < k; p++) {
        C[ j * m + i ] += A[ p * m + i ] * B[ j * k + p ];
      }
    }
  }
}
#endif

#if defined(RUNSTEP1)
void step1_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  // Naive + Transpose(A) GEMM
  for (i = 0; i < m; i++) {
    const auto A_ptr = &A [ i * k /*+ p*/ ];
    for (j = 0; j < n; j++) {
      const auto C_ptr = &C [ j * m /*+ i*/ ];
      const auto B_ptr = &B [ j * k /*+ p*/];
      for (p = 0; p < k; p++) {
        C_ptr[i] += A_ptr[p] * B_ptr[p];
      }
    }
  }
}
#endif

#if defined(RUNSTEP2)
void step2_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  // Naive + Interchange (i,j,p) -> (j,p,i) GEMM
  for (j = 0; j < n; j++) {
    auto B_ptr = &B [ j * k /*+ p*/ ];
    auto C_ptr = &C [ j * m /*+ i*/ ];
    for (p = 0; p < k; p++) {
      auto A_ptr = &A [ p * m /*+ i*/ ];
      for (i = 0; i < m; i++) {
        C_ptr[i] += A_ptr[i] * B_ptr[p];
      }
    }
  }
}
#endif

#if defined(RUNSTEP3)
void step3_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  constexpr int i_blk = 128;
  constexpr int j_blk = 64;
  constexpr int p_blk = 128;

  // Naive + Interchange + Blocking GEMM
  for (j = 0; j < n; j+=j_blk) {
    for (p = 0; p < k; p+=p_blk) {
      for (i = 0; i < m; i+=i_blk) {
        const int jj_end = std::min(n, j + j_blk);
        for (long jj = j; jj < jj_end; jj++ ) {
          auto B_ptr = &B [ jj * k /*+ pp*/ ];
          auto C_ptr = &C [ jj * m /*+ ii*/ ];
          const int pp_end = std::min(k, p + p_blk);
          for (long pp = p; pp < pp_end; pp++) {
            auto A_ptr = &A [ pp * m /*+ ii*/ ];
            const int ii_end = std::min(m, i + i_blk);
            for (long ii = i; ii < ii_end; ii++) {
              C_ptr[ii] += A_ptr[ii] * B_ptr[pp];
            }
          }
        }
      }
    }
  }
}
#endif

#if defined(RUNSTEP4)
void step4_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  int i, j, p;

  constexpr int i_blk = 128;
  constexpr int j_blk = 64;
  constexpr int p_blk = 128;

#define pack_matrix(Mat_pack, Mat, t, t_blk, s_blk, S) \
  for (auto _s = 0,pack_it=0; _s < S; _s+=s_blk) \
    for (auto _t = t; _t < t+t_blk; _t++) { \
      const double *Mat_ptr = &Mat[_t*S]; \
      for (auto _ss = _s; _ss < _s+s_blk; _ss+=8) { \
        Mat_pack[pack_it + 0] = Mat_ptr[_ss + 0]; \
        Mat_pack[pack_it + 1] = Mat_ptr[_ss + 1]; \
        Mat_pack[pack_it + 2] = Mat_ptr[_ss + 2]; \
        Mat_pack[pack_it + 3] = Mat_ptr[_ss + 3]; \
        Mat_pack[pack_it + 4] = Mat_ptr[_ss + 4]; \
        Mat_pack[pack_it + 5] = Mat_ptr[_ss + 5]; \
        Mat_pack[pack_it + 6] = Mat_ptr[_ss + 6]; \
        Mat_pack[pack_it + 7] = Mat_ptr[_ss + 7]; \
        pack_it += 8; \
      } \
    }

  double *A_pack = (double*)alloca(sizeof(double)*p_blk*m);
  double *B_pack = (double*)alloca(sizeof(double)*j_blk*k);

  // Naive + Interchange + Blocking + Packing GEMM
  for (j = 0; j < n; j+=j_blk) {
    int blockB;
    pack_matrix(B_pack, B, j, j_blk, p_blk, k);
    for (p = 0, blockB = 0; p < k; p+=p_blk, blockB++) {
      int blockA;
      pack_matrix(A_pack, A, p, p_blk, i_blk, m);
      for (i = 0, blockA = 0; i < m; i+=i_blk, blockA++) {
        const int jj_end = std::min(n, j + j_blk);
        for (long jj = j, pack_j=0; jj < jj_end; jj++, pack_j++) {
          auto B_ptr = &B_pack [ blockB * (j_blk * p_blk) + pack_j * p_blk];
          auto C_ptr = &C [ jj * m /*+ ii*/ ];
          const int pp_end = std::min(k, p + p_blk);
          for (long pp = p, pack_p=0; pp < pp_end; pp++, pack_p++) {
            auto A_ptr = &A_pack [ blockA * (p_blk * i_blk) + pack_p * i_blk ];
            const int ii_end = std::min(m, i + i_blk);
            for (long ii = i, pack_i=0; ii < ii_end; ii++, pack_i++) {
              C_ptr[ii] += A_ptr[pack_i] * B_ptr[pack_p];
            }
          }
        }
      }
    }
  }
}
#endif

#ifdef DISABLE_VERIFY
void step9_dgemm(double *A, double *B, double *C, int m, int k, int n) { }
#else
template<StorageOptions ALayout, StorageOptions BLayout, StorageOptions CLayout>
void step9_dgemm(double *A, double *B, double *C, int m, int k, int n) {
  DynStride sA(m, 1);
  DynStride sB(k, 1);
  DynStride sC(m, 1);

  Map<Matrix<double, Dynamic, Dynamic, ALayout>, 0, DynStride> eA(A, m, k, sA);
  Map<Matrix<double, Dynamic, Dynamic, BLayout>, 0, DynStride> eB(B, k, n, sB);
  Map<Matrix<double, Dynamic, Dynamic, CLayout>, 0, DynStride> eC(C, m, n, sC);

  eC += eA * eB;
}
#endif

#ifdef RUNBLAS
void blas_dgemm(double *A, double *B, double *C, int m, int k, int n) {

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      m, n, k, 1.0, A, m, B, k, 1.0, C, m);
}
#endif /*RUNBLAS*/

#ifdef RUNBLIS
void blis_dgemm(double *A, double *B, double *C, int m, int k, int n) {

  dim_t _m = m;
  dim_t _k = k;
  dim_t _n = n;

  inc_t rsa = 1;
  inc_t csa = m;

  inc_t rsb = 1;
  inc_t csb = k;

  inc_t rsc = 1;
  inc_t csc = m;

  double alpha = 1.0;
  double beta = 1.0;

  bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
      _m, _n, _k, &alpha, A, rsa, csa, B, rsb, csb, &beta, C, rsc, csc);
}
#endif /*RUNBLIS*/

#ifdef RUNMKL
void mkl_dgemm(double *A, double *B, double *C, int m, int k, int n) {

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      m, n, k, 1.0, A, m, B, k, 1.0, C, m);
}
#endif /*RUNMKL*/

#ifdef RUNESSL
void essl_dgemm(double *A, double *B, double *C, int m, int k, int n) {

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      m, n, k, 1.0, A, m, B, k, 1.0, C, m);
}
#endif /*RUNESSL*/

bool matchResults(const double * const C, const double * const CExp,
    int m, int n) {
  constexpr double RHO = 1.0e-8;
  auto max_diff = std::numeric_limits<double>::min();
  for (int i = 0; i < m * n; ++i) {
    auto diff = std::fabs(C[i] - CExp[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  if (max_diff <= RHO) {
    return true;
  }
  return false;
}

void randomizeMatrix(double *mat, int d1, int d2) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<uint64_t> dist(0, 2);

  for (int i = 0; i < d1 * d2; ++i) {
    mat[i] = static_cast<double>(dist(mt));
  }
}

typedef std::function<void(double*,double*,double*,int,int,int)> GEMM_FUNC;

void runGeMMTest(const char* name, GEMM_FUNC f,
    GEMM_FUNC g, double *A, double *B, double *C, double *CExp, int m, int k,
    int n) {
  randomizeMatrix(A, m, k);
  randomizeMatrix(B, k, n);
  memset(C, 0, sizeof(double)*m*n);
  memset(CExp, 0, sizeof(double)*m*n);

  auto t0 = chrono::high_resolution_clock::now();
  f(A, B, C, m, k, n);
  auto t1 = chrono::high_resolution_clock::now();

  cerr << name << " "
       << chrono::duration_cast<chrono::milliseconds>(t1-t0).count()
       << " ms";

#ifndef DISABLE_VERIFY
  g(A, B, CExp, m, k, n);

  bool matched = matchResults(C, CExp, m, n);

  cerr << " (results match? ";
  if (matched) {
    cerr << "**YES**!)";
  } else {
    cerr << "**NO**!)";
  }
#endif
  cerr << '\n';
}

int main(int argc, char **argv) {

  double *A = nullptr;
  double *B = nullptr;
  double *C = nullptr;
  double *CExp = nullptr;
#define check_alloc(a) \
  if (a != 0) { \
    perror("posix_memalign"); \
    exit(-1); \
  }
  const long PAGE_SIZE = sysconf(_SC_PAGESIZE);
  check_alloc(posix_memalign((void**)&A,    PAGE_SIZE, sizeof(double) * M_DIM * K_DIM));
  check_alloc(posix_memalign((void**)&B,    PAGE_SIZE, sizeof(double) * K_DIM * N_DIM));
  check_alloc(posix_memalign((void**)&C,    PAGE_SIZE, sizeof(double) * M_DIM * N_DIM));
  check_alloc(posix_memalign((void**)&CExp, PAGE_SIZE, sizeof(double) * M_DIM * N_DIM));

#if defined(DISABLE_VERIFY)
#if !defined(RUNSTEP1)
  auto eigen1 = step9_dgemm;
#else
  auto eigen2 = step9_dgemm;
#endif /* RUNSTEP1 */
#else
#if !defined(RUNSTEP1)
  auto eigen1 = step9_dgemm<ColMajor, ColMajor, ColMajor>;
#else
  auto eigen2 = step9_dgemm<RowMajor, ColMajor, ColMajor>;
#endif /* RUNSTEP1 */
#endif /* !DISABLE_VERIFY */

#if defined(WARMUP)
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      M_DIM, N_DIM, K_DIM, 1.0, A, M_DIM, B, K_DIM, 1.0, C, M_DIM);
#endif

#if defined(RUNSTEP0)
  runGeMMTest("Step0", step0_dgemm, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNSTEP1)
  runGeMMTest("Step1", step1_dgemm, eigen2, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNSTEP2)
  runGeMMTest("Step2", step2_dgemm, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNSTEP3)
  runGeMMTest("Step3", step3_dgemm, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNSTEP4)
  runGeMMTest("Step4", step4_dgemm, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNEIGEN)
  runGeMMTest("Eigen",      eigen1, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNBLAS)
  runGeMMTest("BLAS" ,  blas_dgemm, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNBLIS)
  runGeMMTest("BLIS" ,  blis_dgemm, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNMKL)
  runGeMMTest("MKL"  ,   mkl_dgemm, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#elif defined(RUNESSL)
  runGeMMTest("ESSL" ,  essl_dgemm, eigen1, A, B, C, CExp, M_DIM, K_DIM, N_DIM);
#else
#error "No kernel enabled."
#endif /* NO RUN */

  free(A);
  free(B);
  free(C);
  free(CExp);

  return 0;
}
