import numpy as np
import optkit as ok

def linear_program(shape, allow_transpose=True, **options):
    r""" Generate a random linear program

        maximize c^T x
            s.t. Ax <= b

    In graph form, this is

        minimize I(y <= b) + c'Tx
            s.t. y = Ax,

    where I(z; S) is the extended value indicator function that
    takes on values 0 if y \in S, +\infty otherwise.

    A generated as A_ij ~ N(0, 1).
    b generated as b = Av + noise, with

            v_i ~ N(0, 1/n)
            noise_i ~ U[0, 1/10]

    c generated as c = -A^Tu, with u_i ~ U[0, 1] to ensure that the
    problem is bounded.
    """
    m, n = shape
    if allow_transpose and m < n:
        n, m = shape
    assert m >= n, "expect dimensions (m, n) with m > n for LP"

    A = np.random.normal(0, 1, (m, n))
    v = np.random.normal(0, 1./n, n)
    noise = np.random.uniform(0, 0.1, m)
    b = np.dot(A, v) + noise

    u = np.random.uniform(0, 1, m)
    c = -np.dot(A.T, u)

    linear_cone = ok.api.PogsObjective(m, h='IndLe0', b=b)
    cx = ok.api.PogsObjective(n, h='Zero', d=c)
    return A, linear_cone, cx
