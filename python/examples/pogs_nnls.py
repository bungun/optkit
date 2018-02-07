import numpy as np
import optkit as ok

def nonnegative_least_squares(shape, allow_transpose=True, **options):
    r"""Generate a random nonnegative least squares problem

    Given problem data A (m-by-n matrix), b' (m-by-1 vector),
    solve the problem

        minimize |Ax - b'|_2^2
            s.t. x >= 0.

    In graph form, we phrase this as

        miniimize |y - b'|_2^2 + I(x >= 0)
             s.t. y = Ax.

    A generated as A_ij in N(0, 1).
    b generated as b = Av + noise, where

        v_i ~ N(1/n, 1/n)

    and

        noise_i ~ N(0, 1/4).
    """
    m, n = shape
    if allow_transpose and m < n:
        n, m = shape
    assert m >= n, "expect dimensions (m, n) with m > n for basis pursuit"

    A = np.random.normal(0, 1, (m, n))
    v = np.random.normal(1./n, 1./n, n)
    noise = np.random.normal(0, 0.25, m)
    b = np.dot(A, v) + noise

    lsq = ok.api.PogsObjective(m, h='Square', b=b)
    nonneg = ok.api.PogsObjective(n, h='IndGe0')
    return A, lsq, nonneg
