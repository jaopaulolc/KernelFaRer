subroutine gemm(m, n, k, alpha, A, lda, B, ldb, C, ldc)
  integer :: m, n, k
  integer :: lda, ldb, ldc
  real, dimension(m * n) :: C
  real, dimension(m * k) :: A
  real, dimension(k * n) :: B
  real :: alpha
  real :: aa, bb

  do mm = 1, m
    do nn = 1, n
      do i = 1, k
        aa = A(mm + i * lda)
        bb = B(nn + i * ldb)
        C(mm + nn * ldc) = C(mm + nn * ldc) + alpha * aa * bb
      end do
    end do
  end do
end subroutine
