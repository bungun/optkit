from optkit.compat import *

import os
import numpy as np
import numpy.linalg as la
import itertools
import unittest

from optkit.libs import enums
from optkit.libs.anderson import AndersonLibs
from optkit.tests import defs
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances

class AndersonContext(okcctx.CVariableContext):
    def __init__(self, lib, n, lookback):
        aa = lib.anderson_accelerator()
        def build():
            assert NO_ERR(lib.anderson_accelerator_init(aa, n, lookback))
            return aa
        def free():
            return lib.anderson_accelerator_free(aa)
        okcctx.CVariableContext.__init__(self, build, free)

class AndersonLibsTestCase(unittest.TestCase):
    """ Python unit tests for optkit_anderson """

    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = AndersonLibs()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_libs_exist(self):
        libs = []
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            libs.append(self.libs.get(
                    single_precision=single_precision, gpu=gpu))
        self.assertTrue( any(libs) )

class AndersonTestCase(unittest.TestCase):
    """ Python unit tests for optkit_anderson """
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(AndersonLibs())
        self.n = 50
        self.lookback = 3

    def setUp(self):
        self.LIBS_LAYOUTS = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS)

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_accelerator_alloc_free(self):
        n, lookback = self.n, self.lookback

        for lib in self.libs:
            with lib as lib:
                aa = lib.anderson_accelerator()
                try:
                    assert NO_ERR(lib.anderson_accelerator_init(aa, n, lookback))
                    assert isinstance(aa.F, lib.matrix_p)
                    assert isinstance(aa.G, lib.matrix_p)
                    assert isinstance(aa.F_gram, lib.matrix_p)
                    assert isinstance(aa.f, lib.vector_p)
                    assert isinstance(aa.g, lib.vector_p)
                    assert isinstance(aa.diag, lib.vector_p)
                    assert isinstance(aa.alpha, lib.vector_p)
                    assert isinstance(aa.ones, lib.vector_p)
                    assert (aa.mu_regularization == 0.01)
                    assert (aa.iter == 0 )
                finally:
                    assert NO_ERR(lib.anderson_accelerator_free(aa))

    def test_anderson_update_matrices(self):
        n, lookback = self.n, self.lookback
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                F = okcctx.CDenseMatrixContext(lib, n, lookback + 1, order)
                G = okcctx.CDenseMatrixContext(lib, n, lookback + 1, order)
                x = okcctx.CVectorContext(lib, n, random=True)
                g = okcctx.CVectorContext(lib, n, random=True)
                aa = AndersonContext(lib, n, lookback)

                with F, G, x, g, aa as aa:
                    i = int(lookback * np.random.rand(1))

                    assert NO_ERR( lib.anderson_update_F_x(aa, F.c, x.c, i) )
                    F.sync_to_py()
                    assert VEC_EQ( F.py[:, i], - x.py, 1e-7, 1e-7 )

                    assert NO_ERR( lib.anderson_update_F_g(aa, F.c, g.c, i) )
                    F.sync_to_py()
                    assert VEC_EQ( F.py[:, i], g.py - x.py, 1e-7, 1e-7 )

                    assert NO_ERR( lib.anderson_update_G(aa, G.c, g.c, i) )
                    G.sync_to_py()
                    assert VEC_EQ( G.py[:, i], g.py, 1e-7, 1e-7 )

    def test_anderson_set_x0(self):
        n, lookback = self.n, self.lookback
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                F = okcctx.CDenseMatrixContext(lib, n, lookback + 1, order)
                x = okcctx.CVectorContext(lib, n, random=True)
                aa = AndersonContext(lib, n, lookback)

                with F, x, aa as aa:
                    assert NO_ERR( lib.anderson_set_x0(aa, x.c) )
                    assert NO_ERR( lib.matrix_memcpy_am(F.pyptr, aa.F, order) )
                    assert VEC_EQ( F.py[:, 0], - x.py, 1e-7, 1e-7 )

    def test_anderson_gramian(self):
        n, lookback = self.n, self.lookback
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                F = okcctx.CDenseMatrixContext(
                        lib, n, lookback + 1, order, random=True)
                F_gram = okcctx.CDenseMatrixContext(
                        lib, lookback + 1, lookback + 1, order)
                aa = AndersonContext(lib, n, lookback)

                RTOL, ATOL, _, _ = STANDARD_TOLS(lib, n, lookback + 1)

                with F, F_gram, aa as aa:
                    F_gram_calc = F.py.T.dot(F.py)
                    F_gram_calc += np.eye(lookback + 1) * np.sqrt(
                            aa.mu_regularization)

                    assert NO_ERR( lib.anderson_regularized_gram(
                            aa, F.c, F_gram.c, aa.mu_regularization) )
                    F_gram.sync_to_py()
                    assert VEC_EQ( F_gram.py, F_gram_calc, ATOL, RTOL)

    @staticmethod
    def py_anderson_solve(F, mu):
        m = F.shape[1]
        F_gram = F.T.dot(F) + np.sqrt(mu) * np.eye(m)
        alpha = la.solve(F_gram, np.ones(m))
        return alpha / np.sum(alpha)

    def test_anderson_solve(self):
        n, lookback = self.n, self.lookback
        for lib in self.libs:
            with lib as lib:
                order = lib.enums.CblasColMajor
                RTOL, ATOL, _, _ = STANDARD_TOLS(lib, n, lookback + 1)

                F = okcctx.CDenseMatrixContext(
                        lib, n, lookback + 1, order, random=True)
                alpha = okcctx.CVectorContext(lib, lookback + 1)
                aa = AndersonContext(lib, n, lookback)

                with F, alpha, aa as aa:
                    alpha_expect = self.py_anderson_solve(
                            F.py, aa.mu_regularization)
                    assert NO_ERR( lib.anderson_solve(
                            aa, F.c, alpha.c, aa.mu_regularization) )
                    alpha.sync_to_py()
                    assert VEC_EQ( alpha.py, alpha_expect, ATOL, RTOL )

    def test_anderson_mix(self):
        n, lookback = self.n, self.lookback
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, ATOL, _, _ = STANDARD_TOLS(lib, n, lookback + 1, 1)

                G = okcctx.CDenseMatrixContext(
                        lib, n, lookback + 1, order, random=True)
                alpha = okcctx.CVectorContext(lib, lookback + 1, random=True)
                x = okcctx.CVectorContext(lib, n, random=True)
                aa = AndersonContext(lib, n, lookback)

                with G, alpha, x, aa as aa:
                    assert NO_ERR( lib.anderson_mix(aa, G.c, alpha.c, x.c) )
                    x.sync_to_py()
                    assert VEC_EQ(
                            x.py, np.dot(G.py, alpha.py), ATOL, RTOL )

    def test_anderson_accelerate(self):
        n, lookback = self.n, self.lookback
        for lib in self.libs:
            with lib as lib:
                RTOL, ATOL, _, _ = STANDARD_TOLS(lib, n, lookback + 1, 1)
                x = okcctx.CVectorContext(lib, n)
                aa = AndersonContext(lib, n, lookback)

                F = np.zeros((n, lookback + 1))
                G = np.zeros((n, lookback + 1))

                with x, aa as aa:
                    mu = aa.mu_regularization
                    assert NO_ERR( lib.anderson_set_x0(aa, x.c) )

                    # TEST THE BEHAVIOR < LOOKBACK AND >= LOOKBACK
                    for k in xrange(lookback + 5):
                        index = k % (lookback + 1)
                        index_next = (k + 1) % (lookback + 1)

                        xcurr = np.random.rand(n)
                        x.py[:] = xcurr

                        F[:, index] += xcurr
                        G[:, index] = xcurr

                        if k < lookback:
                            xnext = xcurr
                        else:
                            xnext = np.dot(G, self.py_anderson_solve(F, mu))

                        F[:, index_next] = -xnext

                        x.sync_to_c()
                        assert NO_ERR( lib.anderson_accelerate(aa, x.c) )

                        x.sync_to_py()
                        assert VEC_EQ( xnext, x.py, ATOL, RTOL )

    def test_anderson_accelerates_gradient_descent(self):
        """ Test Anderson acceleration for gradient descent on a LS problem

            min. (1/2)||Ax - b||_2^2

            GD:
                x_{k+1} -= alpha_k grad(x_k)
        """
        m = 100
        n, lookback = self.n, self.lookback
        def descent_ls(A, b, x, maxiter=5000, tol=1e-4, accelerate=None):
            def fn(A, b, x):
                residual = A.dot(x) - b
                return residual.dot(residual)

            def linesearch(A, b, x, dx, alpha_):
                armijo = 0.1
                backtrack = 0.1
                step = dx.dot(dx)
                fprev = fn(A, b, x)
                fcurr = fprev
                for i in range(100):
                    fcurr = fn(A, b, x - alpha_ * dx)
                    diff = fprev - fcurr
                    if diff > armijo * alpha_ * step:
                        break
                    alpha_ *= backtrack
                return alpha_

            def grad(A, b, x):
                return A.T.dot(A.dot(x) - b)

            def calc_tol(A, b, x):
                return np.linalg.norm(grad(A, b, x))

            def iterate(A, b, x, alpha_):
                gradx = grad(A, b, x.py)
                alpha_ = linesearch(A, b, x.py, gradx, alpha_)
                x.py -= alpha_ * gradx
                return alpha

            if accelerate is None:
                accelerate = lambda iter: None

            x.py *= 0
            alpha = 1.
            for k in range(maxiter):
                alpha = iterate(A, b, x, alpha)
                accelerate(x)
                if calc_tol(A, b, x.py) < tol:
                    break
            return x, k

        for lib in self.libs:
            with lib as lib:
                TOL = 1e-4

                x = okcctx.CVectorContext(lib, n)
                aa = AndersonContext(lib, n, lookback)
                A = np.random.random((m, n))
                b = np.random.random(m)

                with x, aa as aa:
                    x0, k0 = descent_ls(A, b, x)
                    assert np.linalg.norm(A.T.dot(A .dot(x0.py) - b)) <= TOL

                    def accelerate(x_):
                        x_.sync_to_c()
                        assert NO_ERR( lib.anderson_accelerate(aa, x_.c) )
                        x_.sync_to_py()

                    aa.mu_regularization = 0
                    x1, k1 = descent_ls(A, b, x, accelerate=accelerate)
                    assert np.linalg.norm(A.T.dot(A.dot(x0.py) - b)) <= TOL
                    assert k1 < k0
