import numpy as np
import optkit as ok

def basis_pursuit(shape, allow_transpose=True, **options):
    r""" Generate a random basis pursuit problem

        minimize |x|_1
            s.t. b = Ax.

    In graph form, this is

        minimize I(y == b) + |x|_1
            s.t. y = Ax,

    where I(z; S) is the extended value indicator function that
    takes on values 0 if z \in S, +\infty otherwise.

    A generated as A_ij ~ N(0, 1).
    b generated as b = Av, with

        v_i ~ 0,        with probability p = 0.5
              N(0, 1)   otherwise.
    """
    m, n = shape
    if allow_transpose and m < n:
        n, m = shape
    assert m >= n, "expect dimensions (m, n) with m > n for basis pursuit"

    A = np.random.normal(0, 1, (m, n))
    v = np.random.normal(0, 1./n, n) * (np.random.uniform(0, 1, n) < 0.5)
    b = np.dot(A, v)

    Iy = ok.api.PogsObjective(m, h='IndEq0', b=b)
    x_1 = ok.api.PogsObjective(n, h='Abs')
    return A, Iy, x_1
