void basicSgemm(int m, int n, int k, float alpha, const float **A, const float **B, float beta, float **C )
{
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm][i];
        float b = B[nn][i];
        c += a * b;
      }
      C[mm][nn] = C[mm][nn] * beta + alpha * c;
    }
  }
}
