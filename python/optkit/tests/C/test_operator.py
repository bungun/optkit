from optkit.compat import *

import os
import numpy as np
import itertools
import unittest

from optkit.libs.operator import OperatorLibs
from optkit.tests import defs
from optkit.tests.C import context_managers as okcctx
from optkit.tests.C import statements

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances
CUSTOM_TOLS = statements.custom_tolerances

def operator_initialized(operator_, m, n, OPERATOR_KIND):
    o = operator_
    assert ( o.size1 == m )
    assert ( o.size2 == n )
    assert ( o.kind == OPERATOR_KIND )
    assert ( o.data != 0 )
    assert ( o.apply != 0 )
    assert ( o.adjoint != 0 )
    assert ( o.fused_apply != 0 )
    assert ( o.fused_adjoint != 0 )
    assert ( o.free != 0 )
    return True

def operator_working(lib, operator_, A):
    o = operator_
    m, n = A.shape
    RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n, True)

    alpha = np.random.rand()
    beta = np.random.rand()

    # allocate vectors x, y
    x = okcctx.CVectorContext(lib, n, random=True)
    y = okcctx.CVectorContext(lib, m)

    with x, y:
        # test Ax
        Ax = A.dot(x.py)
        assert NO_ERR( o.apply(o.data, x.c, y.c) )
        y.sync_to_py()
        assert VEC_EQ( y.py, Ax, ATOLM, RTOL )

        # test A'y
        # y.py[:] = Ax[:]   # (update for consistency)
        Aty = np.dot(A.T, y.py)
        assert NO_ERR( o.adjoint(o.data, y.c, x.c) )
        x.sync_to_py()
        assert VEC_EQ( x.py, Aty, ATOLN, RTOL )

        # test Axpy
        # x_[:]  = Aty[:] # (update for consistency)
        Axpy = alpha * A.dot(x.py) + beta * y.py
        assert NO_ERR( o.fused_apply(o.data, alpha, x.c, beta, y.c) )
        y.sync_to_py()
        assert VEC_EQ( y.py, Axpy, ATOLM, RTOL )

        # test A'ypx
        # y_[:] = Axpy[:] # (update for consistency)
        Atypx = alpha * A.T.dot(y.py) + beta * x.py
        assert NO_ERR( o.fused_adjoint(o.data, alpha, y.c, beta, x.c) )
        x.sync_to_py()
        assert VEC_EQ( x.py, Atypx, ATOLN, RTOL )
        return True

def operator_passes(lib, opctx, opkind, A):
    m, n = A.shape
    with opctx as o:
        assert operator_initialized(o.contents, m, n, opkind)
    with opctx as o:
        assert operator_working(lib, o.contents, A)
    return True

class OperatorLibsTestCase(unittest.TestCase):
    """TODO: docstring"""

    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(OperatorLibs())
        self.A_test = dict(
                dense=defs.A_test_gen(),
                sparse=defs.A_test_sparse_gen())

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_OPS = itertools.product(self.libs, defs.OPKEYS)

    def test_libs_exist(self):
        assert any(self.libs)

    def test_dense_alloc_free(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                TOL = 10**(-7 + 2 * lib.FLOAT + 1 * lib.GPU)
                for rowmajor in (True, False):
                    A = defs.A_test_gen()
                    ctx = okcctx.CDenseOperatorContext(lib, A, rowmajor)
                    assert operator_passes(lib, ctx, lib.enums.DENSE, A)

    def test_sparse_operator(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                TOL = 10**(-7 + 2 * lib.FLOAT + 1 * lib.GPU)
                for rowmajor in (True, False):
                    enum = lib.enums.SPARSE_CSR if rowmajor else \
                           lib.enums.SPARSE_CSC
                    A = defs.A_test_sparse_gen()
                    ctx = okcctx.CSparseOperatorContext(lib, A, rowmajor)
                    assert operator_passes(lib, ctx, enum, A)

    def test_diagonal_operator(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                d_ = self.A_test['dense'][0, :]
                ctx = okcctx.CDiagonalOperatorContext(lib, d_)
                assert operator_passes(lib, ctx, lib.enums.DIAGONAL, np.diag(d_))
