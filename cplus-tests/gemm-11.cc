void basicSgemm(int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc )
{
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      for (int i = 0; i < k; ++i) {
        float a = A[mm + i * lda];
        float b = B[nn + i * ldb];
        C[mm+nn*ldc] += alpha * a * b;
      }
    }
  }
}
