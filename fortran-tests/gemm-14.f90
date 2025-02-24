subroutine gemm(m, n, k, A, B, C)
  integer :: m, n, k
  real, dimension(m, n) :: C
  real, dimension(m, k) :: A
  real, dimension(k, n) :: B
  real :: aa, bb, cc

  do mm = 1, m
    do nn = 1, n
      cc = 0
      do i = 1, k
        aa = A(mm, i)
        bb = B(nn, i)
        cc = cc + aa * bb
      end do
      C(mm, nn) = cc
    end do
  end do
end subroutine
