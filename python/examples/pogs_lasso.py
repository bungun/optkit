import numpy as np
import optkit as ok

def lasso(shape, allow_transpose=True, lambdaa=None, **options):
    r""" Generate a random LASSO problem

        minimize |Ax - b|_2^2 + \lambda |x|_1

    In graph form, this is

        minimize |y - b|_2 + \lambda |x|_1
            s.t. y = Ax.

    A generated as A_ij ~ N(0, 1).
    b generated as Av + noise, with

        v_i ~ 0,            with probability p = 0.5
              N(0, 1/n),    otherwise,

    and

        noise_i ~ N(0, 1/4).

    The regularization parameter is chosen to be

        \lambda = (1/5)|A^Tb|_\infty,

    since the solution x^\star = 0 for \lambda > |A^Tb|_\infty.
    """
    m, n = shape
    if allow_transpose and m > n:
        n, m = shape
    assert m <= n, "expect dimensions (m, n) with m < n for LASSO"

    A = np.random.normal(0, 1, (m, n))
    v = np.random.normal(0, 1./n, n) * (np.random.uniform(0, 1, n) < 0.5)
    noise = np.random.normal(0, .25, m)
    b = np.dot(A, v) + noise
    if lambdaa is None:
        lambdaa = 0.2 * np.max(np.dot(A.T, b))
    else:
        lambdaa = float(lambdaa)

    l2_y = ok.api.PogsObjective(m, h='Square', b=b)
    l1_x = ok.api.PogsObjective(n, h='Abs', c=lambdaa)
    return A, l2_y, l1_x
