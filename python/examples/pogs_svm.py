import numpy as np
import optkit as ok

def support_vector_machine(shape, allow_transpose=True, lambdaa=1.,
                           **options):
    r""" Generate a random support vector machine problem

        minimize x^Tx + \lambda sum_i^m max(0, b_ia_i^Tx + 1),

    where b_i \in {-1, +1} is a class label and a_i^T is the ith
    row of matrix A. In graph form, this is

        minimize \lambda max(0, diag(b) * y + 1) + x^T x
            s.t. y = Ax.

    Label vector b chosen such that first m/2 elements belong to the
    first class and remaining elements belong to the second class, i.e.,

        b_i = +1,   i <= m/2
              -1,   otherwise.

    A generated as

        A_ij ~ N(+1/n, 1/n),    i <= m/2
               N(-1/n, 1/n),    otherwise.
    """
    m, n = shape
    if allow_transpose and m < n:
        n, m = shape
    assert m >= n, "expect dimensions (m, n) with m > n for SVM"


    A = np.zeros((m, n))
    A[:m/2, :] = np.random.normal(1./n, 1./n, (m/2, n))
    A[m/2:, :] = np.random.normal(-1./n, 1./n, (m - m/2, n))

    b = np.ones(m)
    b[m/2:] *= -1

    lambdaa = float(lambdaa)

    obj_y = ok.api.PogsObjective(m, h='MaxPos0', a=b, b=-1, c=lambdaa)
    obj_x = ok.api.PogsObjective(n, h='Square')
    return A, obj_y, obj_x
