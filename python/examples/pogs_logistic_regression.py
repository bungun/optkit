import numpy as np
import optkit as ok

def logistic_regression(shape, allow_transpose=True, lambdaa=None,
                        elastic_net=False, alpha=0.5, **options):
    r""" Generate a random logistic regression problem

        minimize \sum_i^m (log(1 + exp(x^Ta_i)) - b_ix^Ta_i) + \lambda |x|_1.

    In graph form, this is

        minimize \sum_i^m (log(1 + exp(y)) - b_iy_i) + \lambda |x|_1
            s.t. y = Ax.

    If `elastic_net=True` is specified as a keyword argument, the
    elastic net regularizer lambda((1-\alpha)|x|_2 + \alpha|x|_1) is
    used instead of the \ell_1 norm only.

    A generated as A_ij ~ N(0, 1).
    b generated as:

        b_i ~ 0, with probability p = 1 / (1 + exp(-a_i^Tv))
              1, otherwise

    for some v generated as

        v_i ~ 0,            with probability p = 0.5
              N(0, 1/n),    otherwise.

    The regularization parameter is chosen to be

        \lambda = (1/10)|A^T((1/2)1 - b)|_\infty,

    since the solution x^\star = 0 for \lambda > |A^T((1/2)1 - b)|_\infty.
    """
    m, n = shape
    if allow_transpose and m < n:
        n, m = shape
    assert m >= n, "expect dimensions (m, n) with m > n for logistic regression"

    A = np.random.normal(0, 1, (m, n))
    v = np.random.normal(0, 1./n, n) * (np.random.uniform(0, 1, n) > 0.5)
    p = np.random.uniform(0, 1, m) < (1. / (1 + np.exp(-np.dot(A, v))))
    b = np.ones(m) *  (1 - p)

    if lambdaa is None:
        lambdaa = 0.1 * np.max(np.dot(A.T, 0.5 - b))
    else:
        lambdaa = float(lambdaa)

    if elastic_net:
        alpha = max(0, min(1, float(alpha)))
        e_x = lambdaa * (1 - alpha)
        lambdaa *= alpha
    else:
        e_x = 0

    loss_y = ok.api.PogsObjective(m, h='Logistic', d=-b)
    reg_x = ok.api.PogsObjective(n, h='Abs', c=lambdaa, e=e_x)
    return A, loss_y, reg_x
