subroutine gemm(m, n, k, A, lda, B, ldb, C, ldc)
  integer :: m, n, k
  integer :: lda, ldb, ldc
  real, dimension(m * n) :: C
  real, dimension(m * k) :: A
  real, dimension(k * n) :: B
  real :: aa, bb, cc

  do mm = 1, m
    do nn = 1, n
      cc = 0
      do i = 1, k
        aa = A(mm + i * lda)
        bb = B(nn + i * ldb)
        cc = cc + aa * bb
      end do
      C(mm + nn * ldc) = cc
    end do
  end do
end subroutine
