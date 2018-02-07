import numpy as np
import optkit as ok

def entropy_maximization(shape, allow_transpose=True, **options):
    r""" Generate a random entropy maximization problem

        maximize -x^T log(x)
            s.t. 1^Tx = 1, Ax <= b.

    In graph form, this is

        minimize I(y <= b) + I(y_s == 1) + x^T log(x)
            s.t. (y, y_s) = (A, 1^T) x,

    where I(z; S) is the extended value indicator function that
    takes on values 0 if z \in S, +\infty otherwise.

    A generated as A_ij ~ N(0, 1).
    b generated as

        b = Av / (1^Tv), v_i ~ U[0, 1]

    to guarantee the existence of a feasible x.
    """
    m, n = shape
    if allow_transpose and m > n:
        n, m = shape
    assert m <= n, "expect dimensions (m, n) with m < n for entropy maximization"

    A = np.random.normal(0, n, (m + 1, n))
    A[-1, :] = 1
    v = np.random.uniform(0, 1, n)
    b = np.dot(A, v) / np.sum(v)

    Iy = ok.api.PogsObjective(m + 1 , h='IndLe0', b=b)
    Iy.set(start=-1, h='IndEq0', b=1)
    x_entr = ok.api.PogsObjective(n, h='NegEntr')
    return A, Iy, x_entr
