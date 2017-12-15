from optkit.compat import *

import os
import numpy as np
import ctypes as ct
import itertools
import unittest

from optkit.libs.cg import ConjugateGradientLibs
from optkit.tests import defs
from optkit.tests.C import statements
from optkit.tests.C import context_managers as okcctx

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances
CUSTOM_TOLS = statements.custom_tolerances

CG_QUIET = 1

class PrecondOpCtx(okcctx.CDiagonalOperatorContext):
    def __init__(self, lib, A, rho):
        n = A.shape[1]
        p = np.zeros(n)
        # calculate diagonal preconditioner
        for j in xrange(n):
            p[j] = 1. / (rho +  np.linalg.norm(A[:, j])**2)
        okcctx.CDiagonalOperatorContext.__init__(self, lib, p)

class CGLSHelperCtx(okcctx.CPointerContext):
    def __init__(self, lib, m, n):
        def alloc_(): return lib.cgls_helper_alloc(m, n)
        okcctx.CPointerContext.__init__(self, alloc_, lib.cgls_helper_free)

class PCGHelperCtx(okcctx.CPointerContext):
    def __init__(self, lib, m, n):
        def alloc_(): return lib.pcg_helper_alloc(m, n)
        okcctx.CPointerContext.__init__(self, alloc_, lib.pcg_helper_free)

def gen_pcg_ctxs(lib, op_, A, rho):
    m, n = A.shape
    T = rho * np.eye(n)
    T += A.T.dot(A)

    o = okcctx.c_operator_context(lib, op_, A)
    p = PrecondOpCtx(lib, T, rho)
    helper = PCGHelperCtx(lib, m, n)
    return o, p, T, helper

def cgls_KKT_converged(A, x, b, rho, flag, tol):
    repeat_factor = 5**(defs.TEST_ITERATE > 0) * 2**(defs.TEST_ITERATE > 1)

    # checks:
    # 1. exit flag == 0
    # 2. KKT condition A'(Ax - b) + rho (x) == 0 (within tol)
    assert ( flag == 0 )
    KKT = A.T.dot(A.dot(x) - b) + rho * x
    if np.linalg.norm(KKT) > tol:
        print('KKT conditions violated:')
        print('norm KKT conditions:', np.linalg.norm(KKT))
        print('tolerance:', tol)
    assert ( np.linalg.norm(KKT) <= repeat_factor * tol )
    return True

def cgls_converged(lib, A, x, b, rho, flag, tol):
    assert ( flag[0] <= lib.CGLS_MAXFLAG )
    x.sync_to_py()
    assert cgls_KKT_converged(A, x.py, b.py, rho, flag[0], tol)
    return True

def pcg_converged(T, x, b, iter_, maxiter, atol, rtol):
    x.sync_to_py()
    assert ( iter_[0] <= maxiter )
    assert VEC_EQ( np.dot(T, x.py), b.py, atol, rtol )
    return True

class ConjugateGradientTestCase(unittest.TestCase):
    """TODO: docstring"""

    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(ConjugateGradientLibs())

        self.tol_cg = 1e-12
        self.rho_cg = 1e-4
        self.maxiter_cg = 1000
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

    def test_cgls_helper_alloc_free(self):
        for lib in self.libs:
            with lib as lib:
                try:
                    h = lib.cgls_helper_alloc(*defs.shape())
                    assert isinstance(h.contents.p, lib.vector_p)
                    assert isinstance(h.contents.q, lib.vector_p)
                    assert isinstance(h.contents.r, lib.vector_p)
                    assert isinstance(h.contents.s, lib.vector_p)
                finally:
                    assert NO_ERR( lib.cgls_helper_free(h) )

    def test_cgls_nonallocating(self):
        """
        cgls_nonallocating test

        given operator A, vector b and scalar rho,
        cgls method attemps to solve

            min. ||Ax - b||_2^2 + rho ||x||_2^2

        to specified tolerance _tol_ by performing at most _maxiter_
        CG iterations on the above least squares problem
        """
        m, n = defs.shape()
        tol = self.tol_cg
        maxiter = self.maxiter_cg
        rho = 1e-2

        # -----------------------------------------
        # test cgls for each operator type defined in defs.OPKEYS
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                ATOLN = n**0.5 * 10**(-7 + 3 * lib.FLOAT)
                TOL = tol * 10**(5 * lib.FLOAT)
                RHO = rho * 10**(1 * lib.FLOAT)

                print('test cgls (nonallocating), operator type:', op_)
                x = okcctx.CVectorContext(lib, n)
                b = okcctx.CVectorContext(lib, m, random=True)
                A = okcctx.c_operator_context(lib, op_, self.A_test[op_])
                h = CGLSHelperCtx(lib, m, n)

                flag = np.zeros(1).astype(ct.c_uint)
                flag_p = flag.ctypes.data_as(ct.POINTER(ct.c_uint))

                with x, b, A as A, h as h:
                    assert NO_ERR( lib.cgls_nonallocating(
                            h, A, b.c, x.c, RHO, TOL, maxiter, CG_QUIET, flag_p) )
                    assert cgls_converged(
                            lib, self.A_test[op_], x, b, RHO, flag, ATOLN)

    def test_cgls_allocating(self):
        tol = self.tol_cg
        maxiter = self.maxiter_cg
        rho = 1e-2

        m, n = defs.shape()
        # -----------------------------------------
        # test cgls for each operator type defined in defs.OPKEYS
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                ATOLN = n**0.5 * 10**(-7 + 3 * lib.FLOAT)
                TOL = tol * 10**(5 * lib.FLOAT)
                RHO = rho * 10**(1 * lib.FLOAT)

                print('test cgls (nonallocating), operator type:', op_)
                x = okcctx.CVectorContext(lib, n)
                b = okcctx.CVectorContext(lib, m, random=True)
                A = okcctx.c_operator_context(lib, op_, self.A_test[op_])

                flag = np.zeros(1).astype(ct.c_uint)
                flag_p = flag.ctypes.data_as(ct.POINTER(ct.c_uint))

                with x, b, A as A:
                    assert NO_ERR( lib.cgls(
                            A, b.c, x.c, RHO, TOL, maxiter, CG_QUIET, flag_p) )
                    assert cgls_converged(
                            lib, self.A_test[op_], x, b, RHO, flag, ATOLN)

    def test_cgls_easy(self):
        tol = self.tol_cg
        maxiter = self.maxiter_cg
        rho = 1e-2

        m, n = defs.shape()
        # -----------------------------------------
        # test cgls for each operator type defined in defs.OPKEYS
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                ATOLN = n**0.5 * 10**(-7 + 3 * lib.FLOAT)
                TOL = tol * 10**(5 * lib.FLOAT)
                RHO = rho * 10**(1 * lib.FLOAT)

                print('test cgls (easy), operator type:', op_)
                x = okcctx.CVectorContext(lib, n)
                b = okcctx.CVectorContext(lib, m, random=True)
                A = okcctx.c_operator_context(lib, op_, self.A_test[op_])

                flag = np.zeros(1).astype(ct.c_uint)
                flag_p = flag.ctypes.data_as(ct.POINTER(ct.c_uint))

                with x, b, A as A, okcctx.CPointerContext(
                        lambda: lib.cgls_init(m, n), lib.cgls_finish) as work:
                    assert NO_ERR( lib.cgls_solve(
                            work, A, b.c, x.c, RHO, TOL, maxiter, CG_QUIET,
                            flag_p) )
                    assert cgls_converged(
                            lib, self.A_test[op_], x, b, RHO, flag, ATOLN)

    def test_pcg_helper_alloc_free(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                try:
                    h = lib.pcg_helper_alloc(*defs.shape())
                    assert isinstance(h.contents.p, lib.vector_p)
                    assert isinstance(h.contents.q, lib.vector_p)
                    assert isinstance(h.contents.r, lib.vector_p)
                    assert isinstance(h.contents.z, lib.vector_p)
                    assert isinstance(h.contents.temp, lib.vector_p)
                finally:
                    assert NO_ERR( lib.pcg_helper_free(h) )

    def test_diagonal_preconditioner(self):
        tol = self.tol_cg
        rho = 1e-2
        maxiter = self.maxiter_cg

        m, n = defs.shape()
        # -----------------------------------------
        # test pcg for each operator type defined in defs.OPKEYS
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                RTOL = 2e-2
                ATOLN = RTOL * n**0.5

                print('test pcg (nonallocating), operator type:', op_)
                A_test = self.A_test[op_]
                A = okcctx.c_operator_context(lib, op_, A_test)
                p = okcctx.CVectorContext(lib, n)

                T = rho * np.eye(n)
                T += A_test.T.dot(A_test)

                # calculate diagonal preconditioner
                p_expect = np.zeros(n)
                for j in xrange(n):
                    p_expect[j] = 1. / (rho +  np.linalg.norm(T[:, j])**2)

                with A as A, p:
                    assert NO_ERR( lib.diagonal_preconditioner(A, p.c, rho) )
                    p.sync_to_py()
                    assert VEC_EQ( p.py, p_expect, ATOLN, RTOL )

    def test_pcg_nonallocating(self):
        """
        pcg_nonallocating test

        given operator A, vector b, preconditioner M and scalar rho,
        pcg method attemps to solve

            (rho * I + A'A)x = b

        to specified tolerance _tol_ by performing at most _maxiter_
        CG iterations on the system

            M(rho * I + A'A)x = b
        """
        tol = self.tol_cg
        maxiter = self.maxiter_cg
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                RTOL, _, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 1, 5, 7)
                RHO = self.rho_cg * 10**(4 * lib.FLOAT)

                print('test pcg (nonallocating), operator type:', op_)
                x = okcctx.CVectorContext(lib, n)
                b = okcctx.CVectorContext(lib, n, random=True)
                A, p, T, h = gen_pcg_ctxs(lib, op_, self.A_test[op_], RHO)

                iter_ = np.zeros(1).astype(ct.c_uint)
                iter_p = iter_.ctypes.data_as(ct.POINTER(ct.c_uint))

                with x, b, A as A, p as p, h as h:
                    assert NO_ERR( lib.pcg_nonallocating(
                            h, A, p, b.c, x.c, RHO, tol, maxiter, CG_QUIET,
                            iter_p) )
                    assert pcg_converged(T, x, b, iter_, maxiter, ATOLN, RTOL)

    def test_pcg_nonallocating_warmstart(self):
        """test pcg warmstart for each operator type defined in defs.OPKEYS"""
        tol = self.tol_cg
        maxiter = self.maxiter_cg
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                RTOL, _, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 1, 5, 7)
                RHO = self.rho_cg * 10**(4 * lib.FLOAT)

                print('test pcg (nonallocating) warmstart, operator type:', op_)
                x = okcctx.CVectorContext(lib, n)
                b = okcctx.CVectorContext(lib, n, random=True)
                A, p, T, h = gen_pcg_ctxs(lib, op_, self.A_test[op_], RHO)

                iter_ = np.zeros(1).astype(ct.c_uint)
                iter_p = iter_.ctypes.data_as(ct.POINTER(ct.c_uint))

                with x, b, A as A, p as p, h as h:
                    assert NO_ERR( lib.pcg_nonallocating(
                            h, A, p, b.c, x.c, RHO, tol, maxiter, CG_QUIET,
                            iter_p) )
                    assert pcg_converged(T, x, b, iter_, maxiter, ATOLN, RTOL)

                    # first run
                    assert NO_ERR( lib.pcg_nonallocating(
                            h, A, p, b.c, x.c, RHO, tol, maxiter, CG_QUIET,
                            iter_p) )
                    assert pcg_converged(T, x, b, iter_, maxiter, ATOLN, RTOL)
                    iters1 = iter_[0]

                    # second run
                    assert NO_ERR( lib.pcg_nonallocating(
                            h, A, p, b.c, x.c, RHO, tol, maxiter, CG_QUIET,
                            iter_p) )
                    assert pcg_converged(T, x, b, iter_, maxiter, ATOLN, RTOL)
                    iters2 = iter_[0]

                    print('cold start iters:', iters1)
                    print('warm start iters:', iters2)
                    assert (iters1 <= maxiter)
                    assert (iters2 <= maxiter)
                    assert (iters2 <= iters1)


    def test_pcg_allocating(self):
        """test pcg (allocating) for each operator type defined in defs.OPKEYS
        """
        tol = self.tol_cg
        maxiter = self.maxiter_cg
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                RTOL, _, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 1, 5, 7)
                RHO = self.rho_cg * 10**(4 * lib.FLOAT)

                print('test pcg (allocating), operator type:', op_)
                x = okcctx.CVectorContext(lib, n)
                b = okcctx.CVectorContext(lib, n, random=True)
                A, p, T, _ = gen_pcg_ctxs(lib, op_, self.A_test[op_], RHO)

                iter_ = np.zeros(1).astype(ct.c_uint)
                iter_p = iter_.ctypes.data_as(ct.POINTER(ct.c_uint))

                with x, b, A as A, p as p:
                    assert NO_ERR( lib.pcg(
                            A, p, b.c, x.c, RHO, tol, maxiter, CG_QUIET, iter_p) )
                    assert pcg_converged(T, x, b, iter_, maxiter, ATOLN, RTOL)

    def test_pcg_easy(self):
        """test pcg (easy) for each operator type defined in defs.OPKEYS"""
        tol = self.tol_cg
        maxiter = self.maxiter_cg
        m, n = defs.shape()
        for lib, op_ in self.LIBS_OPS:
            with lib as lib:
                RTOL, _, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 1, 5, 7)
                RHO = self.rho_cg * 10**(4 * lib.FLOAT)

                print('test pcg (easy), operator type:', op_)
                x = okcctx.CVectorContext(lib, n)
                b = okcctx.CVectorContext(lib, n, random=True)
                A, p, T, _ = gen_pcg_ctxs(lib, op_, self.A_test[op_], RHO)

                iter_ = np.zeros(1).astype(ct.c_uint)
                iter_p = iter_.ctypes.data_as(ct.POINTER(ct.c_uint))

                with x, b, A as A, p as p, okcctx.CPointerContext(
                        lambda: lib.pcg_init(m, n), lib.pcg_finish) as work:

                    assert NO_ERR( lib.pcg_solve(
                            work, A, p, b.c, x.c, RHO, tol, maxiter, CG_QUIET,
                            iter_p) )
                    iters1 = iter_[0]
                    assert pcg_converged(T, x, b, iter_, maxiter, ATOLN, RTOL)

                    assert NO_ERR( lib.pcg_solve(
                            work, A, p, b.c, x.c, RHO, tol, maxiter, CG_QUIET,
                            iter_p) )
                    iters2 = iter_[0]
                    assert pcg_converged(T, x, b, iter_, maxiter, ATOLN, RTOL)

                    print('cold start iters:', iters1)
                    print('warm start iters:', iters2)
                    assert (iters2 <= iters1)
