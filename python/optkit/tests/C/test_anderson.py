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

class GradientDescent:
    def __init__(self, loss, gradient, condition):
        self.loss = loss
        self.gradient = gradient
        self.condition = condition

    def linesearch(self, loss, x, dx, stepsize, backtrack=0.1,
                   armijo=0.1, maxiter_linesearch=100, **options):
        step = armijo * np.dot(dx, dx)
        fprev = loss(x)
        for i in range(maxiter_linesearch):
            if fprev - loss(x - stepsize * dx) > stepsize * step:
                break
            stepsize *= backtrack
        return stepsize

    def __call__(self, x, stepsize=1., tol=1e-4, maxiter=50000, **options):
        self.accelerate = options.pop('accelerate', lambda x: None)
        verbose = options.pop('verbose', False)
        print_iter = options.pop('print_iter', 100)

        x.py *= 0
        for k in range(maxiter):
            gradx = self.gradient(x.py)
            stepsize = self.linesearch(self.loss, x.py, gradx, stepsize, **options)
            x.py -= stepsize * gradx
            self.accelerate(x)
            if verbose and k % print_iter == 0:
                print '{:8}{:20}'.format(k, self.condition(x.py, gradx))
            if self.condition(x.py, gradx) < tol:
                break
        return x, k

class AndersonContext(okcctx.CVariableContext):
    def __init__(self, lib, n, lookback):
        aa = lib.anderson_accelerator()
        def build():
            assert NO_ERR(lib.anderson_accelerator_init(aa, n, lookback))
            return aa
        def free():
            return lib.anderson_accelerator_free(aa)
        okcctx.CVariableContext.__init__(self, build, free)

class AndersonDifferenceContext(okcctx.CVariableContext):
    def __init__(self, lib, n, lookback):
        aa = lib.difference_accelerator()
        def build():
            assert (
                lib.anderson_difference_accelerator_init(aa, n, lookback) == 0)
            return aa
        def free():
            return lib.anderson_difference_accelerator_free(aa)
        okcctx.CVariableContext.__init__(self, build, free)

class AndersonFusedContext(okcctx.CVariableContext):
    def __init__(self, lib, n, lookback):
        aa = lib.fused_accelerator()
        def build():
            assert (
                lib.anderson_fused_accelerator_init(aa, n, lookback) == 0)
            return aa
        def free():
            return lib.anderson_fused_accelerator_free(aa)
        okcctx.CVariableContext.__init__(self, build, free)

class AndersonFusedDiffContext(okcctx.CVariableContext):
    def __init__(self, lib, n, lookback):
        aa = lib.fused_diff_accelerator()
        def build():
            assert (
                lib.anderson_fused_diff_accelerator_init(aa, n, lookback) == 0)
            return aa
        def free():
            return lib.anderson_fused_diff_accelerator_free(aa)
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
                    assert isinstance(aa.alpha, lib.vector_p)
                    assert isinstance(aa.ones, lib.vector_p)
                    assert (aa.mu_regularization == 0.01)
                    assert (aa.iter == 0 )
                finally:
                    assert NO_ERR(lib.anderson_accelerator_free(aa))

    def test_accelerator_alternate_alloc_free(self):
        n, lookback = self.n, self.lookback

        for lib in self.libs:
            with lib as lib:
                aa = lib.difference_accelerator()
                try:
                    assert NO_ERR(lib.anderson_difference_accelerator_init(
                        aa, n, lookback))
                    assert isinstance(aa.DX, lib.matrix_p)
                    assert isinstance(aa.DF, lib.matrix_p)
                    assert isinstance(aa.DG, lib.matrix_p)
                    assert isinstance(aa.DXDF, lib.matrix_p)
                    assert isinstance(aa.f, lib.vector_p)
                    assert isinstance(aa.g, lib.vector_p)
                    assert isinstance(aa.x, lib.vector_p)
                    assert isinstance(aa.alpha, lib.vector_p)
                    assert isinstance(aa.pivot, lib.int_vector_p)
                    assert (aa.iter == 0 )
                finally:
                    assert NO_ERR(lib.anderson_difference_accelerator_free(aa))

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

                    assert NO_ERR( lib.anderson_update_F_x(F.c, x.c, i) )
                    F.sync_to_py()
                    assert VEC_EQ( F.py[:, i], - x.py, 1e-7, 1e-7 )

                    assert NO_ERR( lib.anderson_update_F_g(F.c, g.c, i) )
                    F.sync_to_py()
                    assert VEC_EQ( F.py[:, i], g.py - x.py, 1e-7, 1e-7 )

                    assert NO_ERR( lib.anderson_update_G(G.c, g.c, i) )
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
                mu = 0.1

                RTOL, ATOL, _, _ = STANDARD_TOLS(lib, n, lookback + 1)

                with F, F_gram, aa as aa:
                    F_gram_calc = F.py.T.dot(F.py)
                    assert NO_ERR( lib.anderson_regularized_gram(
                            aa.linalg_handle, F.c, F_gram.c, 0) )
                    F_gram.sync_to_py()
                    assert VEC_EQ( F_gram.py, F_gram_calc, ATOL, RTOL)

                    assert NO_ERR( lib.anderson_regularized_gram(
                            aa.linalg_handle, F.c, F_gram.c, mu) )
                    F_gram.sync_to_py()
                    mu_eff = np.sqrt(mu) * np.max(np.diag(F_gram_calc))
                    F_gram_calc += np.eye(lookback + 1) * mu_eff
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
                    alpha_expect = self.py_anderson_solve(F.py, 0.)
                    assert NO_ERR( lib.anderson_solve(
                            aa.linalg_handle, F.c, aa.F_gram, alpha.c, aa.ones,
                            aa.mu_regularization) )
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
                    assert NO_ERR( lib.anderson_mix(
                            aa.linalg_handle, G.c, alpha.c, x.c) )
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
                            xnext = np.dot(G, self.py_anderson_solve(F, 0))

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
        pass

        m = 100
        n, lookback = self.n, self.lookback

        for lib in self.libs:
            with lib as lib:
                TOL = 1e-4

                x = okcctx.CVectorContext(lib, n)
                aa = AndersonContext(lib, n, lookback)
                A = np.random.random((m, n))
                b = np.random.random(m)

                def least_squares(x_):
                    residual = A.dot(x_) - b
                    return residual.dot(residual)
                def grad_LS(x_): return A.T.dot(A.dot(x_) - b)
                def tol_LS(x_, dx): return np.linalg.norm(dx)
                gradient_descent = GradientDescent(least_squares, grad_LS, tol_LS)

                with x, aa as aa:
                    x0, k0 = gradient_descent(x, tol=TOL)
                    assert np.linalg.norm(A.T.dot(A.dot(x0.py) - b)) <= TOL

                    def accelerate(x_):
                        x_.sync_to_c()
                        assert NO_ERR( lib.anderson_accelerate(aa, x_.c) )
                        x_.sync_to_py()

                    aa.mu_regularization = 0
                    x1, k1 = gradient_descent(x, accelerate=accelerate)
                    assert np.linalg.norm(A.T.dot(A.dot(x1.py) - b)) <= TOL
                    assert k1 < k0

    def test_anderson_difference_accelerates_gradient_descent(self):
        """ Test Anderson acceleration for gradient descent on a LS problem

            min. (1/2)||Ax - b||_2^2

            GD:
                x_{k+1} -= alpha_k grad(x_k)
        """

        m = 100
        n, lookback = 50, 20
        for lib in self.libs:
            with lib as lib:
                TOL = 1e-4

                x = okcctx.CVectorContext(lib, n)
                aa = AndersonDifferenceContext(lib, n, lookback)
                A = np.random.random((m, n))
                b = np.random.random(m)

                def least_squares(x_):
                    residual = A.dot(x_) - b
                    return residual.dot(residual)
                def grad_LS(x_): return A.T.dot(A.dot(x_) - b)
                def tol_LS(x_, dx): return np.linalg.norm(dx)

                with x, aa as aa:
                    gradient_descent = GradientDescent(least_squares, grad_LS, tol_LS)
                    x0, k0 = gradient_descent(x, tol=TOL)
                    assert np.linalg.norm(A.T.dot(A.dot(x0.py) - b)) <= TOL

                    def accelerate(x_):
                        x_.sync_to_c()
                        assert NO_ERR( lib.anderson_difference_accelerate(aa, x_.c) )
                        x_.sync_to_py()

                    x1, k1 = gradient_descent(x, accelerate=accelerate)
                    assert np.linalg.norm(A.T.dot(A.dot(x1.py) - b)) <= TOL
                    assert k1 < k0
