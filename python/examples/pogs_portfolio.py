import numpy as np
import optkit as ok

def portfolio_optimization(shape, allow_transpose=True, gamma=1., **options):
    r""" Generate a random portfolio optimization problem

    Maximize risk-adjusted return of a portfolio. k-factor risk
    model, return covariance matrix is diagonal + rank k.

        maximize \mu^T x - \gamma x^T(FF^T + D)x
            s.t. x >= 0, 1'x = 1

    Here, F \in \reals^{n \times k}. In graph form, this is

    minimize -\mu^Tx + \gamma x^TDx + I(x >= 0) + \gamma y^Ty + I(y_s == 1)
        s.t. (y y_s) = (F^T, 1^T) x

    where I(z; S) is the extended value indicator function that
    takes on values 0 if z \in S, +\infty otherwise.

    F is generated as F_ij ~ N(0, 1).
    Diagonal of D generated as D_ii ~ U[0, \sqrt{k}]
    Mean return \mu is generated as \mu_i ~ N(0, 1).

    Risk aversion factor \gamma set to 1 by default.
    """
    k, n = shape
    if allow_transpose and k > n:
        n, k = shape
    assert k <= n, "expect dimensions (k, n) with k < n for portfolio optimization"

    F = np.random.normal(0, 1, (n, k))
    A = np.ones((k + 1, n))
    A[:k, :] += F.T

    D_diag = np.random.uniform(0, np.sqrt(k))
    mu = np.random.normal(0, 1, n)
    gamma = max(float(gamma), 0)

    obj_y = ok.api.PogsObjective(k + 1, h='Square', c=gamma)
    obj_y.set(start=-1, h='IndEq0', b=1, c=1)
    obj_x = ok.api.PogsObjective(n, h='IndGe0', d=-mu, e=gamma * D_diag)
    return A, obj_y, obj_x
