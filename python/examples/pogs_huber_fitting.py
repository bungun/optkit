import numpy as np
import optkit as ok

def huber_fitting(shape, allow_transpose=True, **options):
    r""" Generate a random Huber fitting problem

        minimize huber(b - Ax).

    In graph form, this is

        minimize huber(b - y) + zero(x)
            s.t. y = Ax.

    A generated as A_ij ~ N(0, n).
    b generated as Av + noise, with

        v_i ~ N(0, 1/n),

    and

        noise_i ~ N(0, 1/4),    with probability p = 0.95
                  U[0, 10],     otherwise.
    """
    m, n = shape
    if allow_transpose and m < n:
        n, m = shape
    assert m >= n, "expect dimensions (m, n) with m > n for Huber fitting"

    A = np.random.normal(0, n, (m, n))
    v = np.random.normal(0, 1./n, n)
    p = np.random.uniform(0, 1, m) < 0.95
    noise = np.random.normal(0, .25, m) * p + np.random.uniform(0, 10, m) * (1-p)
    b = np.dot(A, v) + noise

    huber_y = ok.api.PogsObjective(m, h='Huber', a=-1, b=-b)
    zero_x = ok.api.PogsObjective(n, h='Zero')
    return A, huber_y, zero_x
