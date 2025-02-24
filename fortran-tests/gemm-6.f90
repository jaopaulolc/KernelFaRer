subroutine gemm(m, n, k, alpha, A, B, C)
  integer :: m, n, k
  real, dimension(m, n) :: C
  real, dimension(m, k) :: A
  real, dimension(k, n) :: B
  real :: alpha
  real :: aa, bb

  do mm = 1, m
    do nn = 1, n
      do i = 1, k
        aa = A(mm, i)
        bb = B(nn, i)
        C(mm, nn) = C(mm, nn) + alpha * aa * bb
      end do
    end do
  end do
end subroutine
