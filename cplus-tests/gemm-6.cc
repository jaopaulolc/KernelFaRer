void basicSgemm(int m, int n, int k, float alpha, const float **A, const float **B, float **C)
{
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      for (int i = 0; i < k; ++i) {
        float a = A[mm][i];
        float b = B[nn][i];
        C[mm][nn] += alpha * a * b;
      }
    }
  }
}
