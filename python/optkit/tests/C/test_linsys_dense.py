from optkit.compat import *

import os
import numpy as np
import ctypes as ct
import itertools
import unittest

from optkit.libs.linsys import DenseLinsysLibs
from optkit.libs import enums
from optkit.tests import defs
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances
CUSTOM_TOLS = statements.custom_tolerances

class DenseLibsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = DenseLinsysLibs()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_libs_exist(self):
        libs = []
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            libs.append(self.libs.get(
                    single_precision=single_precision, gpu=gpu))
        assert any(libs)

    def test_lib_types(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                continue

            assert ( 'ok_float' in dir(lib) )
            assert ( 'ok_int' in dir(lib) )
            assert ( 'c_int_p' in dir(lib) )
            assert ( 'ok_float_p' in dir(lib) )
            assert ( 'ok_int_p' in dir(lib) )
            assert ( 'vector' in dir(lib) )
            assert ( 'vector_p' in dir(lib) )
            assert ( 'matrix' in dir(lib) )
            assert ( 'matrix_p' in dir(lib) )
            assert ( single_precision == (lib.ok_float == ct.c_float) )
            assert ( lib.ok_int == ct.c_int )

    def test_blas_handle(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                continue

            handle = ct.c_void_p()
            # create
            assert NO_ERR( lib.blas_make_handle(ct.byref(handle)) )
            # destroy
            assert NO_ERR( lib.blas_destroy_handle(handle) )

    def test_lapack_handle(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                    continue
            if gpu:
                continue
                # TODO: GPU IMPLEMENTATION

            handle = ct.c_void_p()
            # create
            assert NO_ERR( lib.lapack_make_handle(ct.byref(handle)) )
            # destroy
            assert NO_ERR( lib.lapack_destroy_handle(handle) )

    def test_device_reset(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                continue

            # reset
            assert NO_ERR( lib.ok_device_reset() )

            # allocate - deallocate - reset
            handle = ct.c_void_p()
            assert NO_ERR( lib.blas_make_handle(ct.byref(handle)) )
            assert NO_ERR( lib.blas_destroy_handle(handle) )
            assert NO_ERR( lib.ok_device_reset() )

    def test_version(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                continue

            major = ct.c_int()
            minor = ct.c_int()
            change = ct.c_int()
            status = ct.c_int()

            lib.optkit_version(
                    ct.byref(major), ct.byref(minor), ct.byref(change),
                    ct.byref(status))

            version = defs.version_string(major.value, minor.value,
                                          change.value, status.value)

            assert ( version != '0.0.0' )
            if defs.VERBOSE_TEST:
                print("denselib version", version)

class DenseBLASTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(DenseLinsysLibs())
        self.A_test = defs.A_test_gen()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_LAYOUTS = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS)

    def test_blas1_dot(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                TOL, _, _, _ = STANDARD_TOLS(lib, m, n)

                hdl = okcctx.CDenseLinalgContext(lib)
                v = okcctx.CVectorContext(lib, m, random=True)
                w = okcctx.CVectorContext(lib, m, random=True)
                with v, w, hdl as hdl:
                    answer, answer_p = okcctx.gen_py_vector(lib, 1)
                    assert NO_ERR( lib.blas_dot(hdl, v.c, w.c, answer_p) )
                    assert SCAL_EQ( answer[0], v.py.dot(w.py), TOL )

    def test_blas1_nrm2(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                TOL, _, _, _ = STANDARD_TOLS(lib, m, n)

                hdl = okcctx.CDenseLinalgContext(lib)
                v = okcctx.CVectorContext(lib, m, random=True)
                with v, hdl as hdl:
                    answer, answer_p = okcctx.gen_py_vector(lib, 1)
                    assert NO_ERR( lib.blas_nrm2(hdl, v.c, answer_p) )
                    assert SCAL_EQ( answer[0], np.linalg.norm(v.py), TOL )

    def test_blas1_asum(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                TOL, _, _, _ = STANDARD_TOLS(lib, m, n)

                hdl = okcctx.CDenseLinalgContext(lib)
                v = okcctx.CVectorContext(lib, m, random=True)
                with v, hdl as hdl:
                    answer, answer_p = okcctx.gen_py_vector(lib, 1)
                    assert NO_ERR( lib.blas_asum(hdl, v.c, answer_p) )
                    assert SCAL_EQ( answer[0], np.linalg.norm(v.py, 1), TOL )

    def test_blas1_scal(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

                hdl = okcctx.CDenseLinalgContext(lib)
                v = okcctx.CVectorContext(lib, m, random=True)
                alpha = np.random.random()

                with v, hdl as hdl:
                    v_expect = alpha * v.py
                    assert NO_ERR( lib.blas_scal(hdl, alpha, v.c) )
                    v.sync_to_py()
                    assert VEC_EQ( v.py, v_expect, ATOLM, RTOL )

    def test_blas1_axpy(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

                hdl = okcctx.CDenseLinalgContext(lib)
                v = okcctx.CVectorContext(lib, m, random=True)
                w = okcctx.CVectorContext(lib, m, random=True)
                with v, w, hdl as hdl:
                    alpha = np.random.random()
                    pyresult = alpha * v.py + w.py
                    assert NO_ERR( lib.blas_axpy(hdl, alpha, v.c, w.c) )
                    w.sync_to_py()
                    assert VEC_EQ( w.py, pyresult, ATOLM, RTOL )

    def test_blas2_gemv(self):
        m, n = defs.shape()
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n)

                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, m, random=True)

                alpha = -0.5 + np.random.random()
                beta = -0.5 + np.random.random()

                with A, x, y, hdl as hdl:
                    # perform y = alpha * A * x + beta *  y
                    pyresult = alpha * A.py.dot(x.py) + beta * y.py
                    assert NO_ERR( lib.blas_gemv(
                            hdl, lib.enums.CblasNoTrans, alpha, A.c, x.c,
                            beta, y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( y.py, pyresult, ATOLM, RTOL )

                    # perform x = alpha * A' * y + beta * x
                    y.py[:] = pyresult[:]
                    pyresult = alpha * A.py.T.dot(y.py) + beta * x.py
                    assert NO_ERR( lib.blas_gemv(
                            hdl, lib.enums.CblasTrans, alpha, A.c, y.c, beta,
                            x.c) )
                    x.sync_to_py()
                    assert VEC_EQ( x.py, pyresult, ATOLN, RTOL )

    def test_blas2_trsv(self):
        m, n = defs.shape()

        # generate lower triangular matrix L
        L_test = self.A_test.T.dot(self.A_test)

        # normalize L so inversion doesn't blow up
        L_test /= np.linalg.norm(L_test)


        for i in xrange(n):
            # diagonal entries ~ 1 to keep condition number reasonable
            L_test[i, i] /= 10**np.log(n)
            L_test[i, i] += 1
            # upper triangle = 0
            for j in xrange(n):
                if j > i:
                    L_test[i, j] *= 0

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, _, ATOLN, _ = STANDARD_TOLS(lib, m, n)

                hdl = okcctx.CDenseLinalgContext(lib)
                L = okcctx.CDenseMatrixContext(lib, n, n, order, value=L_test)
                x = okcctx.CVectorContext(lib, n, random=True)

                with L, x, hdl as hdl:
                    # y = inv(L) * x
                    pyresult = np.linalg.solve(L_test, x.py)
                    assert NO_ERR( lib.blas_trsv(hdl, lib.enums.CblasLower,
                                  lib.enums.CblasNoTrans, lib.enums.CblasNonUnit,
                                  L.c, x.c) )
                    x.sync_to_py()
                    assert VEC_EQ( x.py, pyresult, ATOLN, RTOL )

    def test_blas2_sbmv(self):
        m, n = defs.shape()
        diags = max(1, min(4, min(m, n) - 1))

        for lib in self.libs:
            with lib as lib:
                RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

                # make diagonal "matrix" D stored as vector d,
                hdl = okcctx.CDenseLinalgContext(lib)
                s = okcctx.CVectorContext(lib, n * diags, random=True)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, n, random=True)

                with s, x, y, hdl as hdl:
                    # y = alpha
                    alpha = np.random.random()
                    beta = np.random.random()
                    pyresult = np.zeros(n)
                    for d in xrange(diags):
                        for j in xrange(n - d):
                            if d > 0:
                                pyresult[d + j] += s.py[d + diags * j] * x.py[j]
                            pyresult[j] += s.py[d + diags * j] * x.py[d + j]
                    pyresult *= alpha
                    pyresult += beta * y.py

                    assert NO_ERR( lib.blas_sbmv(hdl, lib.enums.CblasColMajor,
                                     lib.enums.CblasLower, diags - 1, alpha, s.c,
                                     x.c, beta, y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( y.py, pyresult, ATOLM, RTOL )

    def test_diagmv(self):
        m, n = defs.shape()

        for lib in self.libs:
            with lib as lib:
                RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

                # make diagonal "matrix" D stored as vector d,
                hdl = okcctx.CDenseLinalgContext(lib)
                d = okcctx.CVectorContext(lib, n, random=True)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, n, random=True)

                with d, x, y, hdl as hdl:
                    # y = alpha * D * x + beta * y
                    alpha = np.random.random()
                    beta = np.random.random()
                    pyresult = alpha * d.py * x.py + beta * y.py
                    assert NO_ERR( lib.blas_diagmv(hdl, alpha, d.c, x.c, beta, y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( y.py, pyresult, ATOLM, RTOL )


    def test_blas3_gemm(self):
        m, n = defs.shape()

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, _, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 1)

                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)
                B = okcctx.CDenseMatrixContext(lib, m, n, order, random=True)
                C = okcctx.CDenseMatrixContext(lib, n, n, order, random=True)

                with A, B, C, hdl as hdl:
                # populate
                    # perform C = alpha * B'A + beta * C
                    alpha = np.random.random()
                    beta = np.random.random()
                    pyresult = alpha * B.py.T.dot(A.py) + beta * C.py
                    assert NO_ERR( lib.blas_gemm(
                            hdl, lib.enums.CblasTrans, lib.enums.CblasNoTrans,
                            alpha, B.c, A.c, beta, C.c) )
                    C.sync_to_py()
                    assert VEC_EQ( C.py, pyresult, ATOLN * 2**0.5, RTOL )


    def test_blas3_syrk(self):
        m, n = defs.shape()
        B_test = np.random.rand(n, n)

        # make B symmetric
        B_test = B_test.T.dot(B_test)

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, _, ATOLN, _ = STANDARD_TOLS(lib, m, n)

                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)
                B = okcctx.CDenseMatrixContext(lib, n, n, order, value=B_test)

                with A, B, hdl as hdl:
                    # B = alpha * (A'A) + beta * B
                    alpha = np.random.random()
                    beta = np.random.random()
                    pyresult = alpha * A.py.T.dot(A.py) + beta * B.py
                    assert NO_ERR( lib.blas_syrk(
                            hdl, lib.enums.CblasLower, lib.enums.CblasTrans,
                            alpha, A.c, beta, B.c) )
                    B.sync_to_py()
                    for i in xrange(n):
                        for j in xrange(n):
                            if j > i:
                                pyresult[i, j] *= 0
                                B.py[i, j] *= 0
                    assert VEC_EQ( B.py, pyresult, ATOLN * 2**0.5, RTOL )


    def test_blas3_trsm(self):
        m, n = defs.shape()

        # make square, invertible L
        L_test = np.random.random((n, n))
        for i in xrange(n):
            L_test[i, i] /= 10**np.log(n)
            L_test[i, i] += 1
            for j in xrange(n):
                if j > i:
                    L_test[i, j]*= 0

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                if lib.GPU:
                    continue

                RTOL, ATOL, _, ATOLMN = CUSTOM_TOLS(lib, m, n, 1, 2)
                repeat_factor = min(4, 2**defs.TEST_ITERATE)
                RTOL *= repeat_factor
                ATOLMN *= repeat_factor

                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)
                L = okcctx.CDenseMatrixContext(lib, n, n, order)

                with A, L, hdl as hdl:
                    L.py += L_test
                    L.sync_to_c()

                    # A = A * inv(L)
                    pyresult = A.py.dot(np.linalg.inv(L_test))
                    assert NO_ERR( lib.blas_trsm(
                            hdl, lib.enums.CblasRight, lib.enums.CblasLower,
                            lib.enums.CblasNoTrans, lib.enums.CblasNonUnit,
                            1., L.c, A.c) )
                    A.sync_to_py()
                    assert VEC_EQ( A.py, pyresult, ATOLMN, RTOL )


class DenseLapackTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(DenseLinsysLibs())
        self.A_test = defs.A_test_gen()
        self.B_test = defs.A_test_gen()
        mindim = min(defs.shape())
        self.eigs = 1 + np.random.uniform(0, 1, mindim)
        self.Q_test, _ = np.linalg.qr(self.A_test[:mindim, :mindim])

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_LAYOUTS = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS)

    def test_vector_LU(self):
        m, n = defs.shape()
        mindim = min(m, n)

        # build decently conditioned square matrix
        C_test = self.A_test.T.dot(self.B_test)[:mindim, :mindim]
        C_test += np.eye(mindim)

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                imprecision_factor = 5**(int(lib.GPU) + int(lib.FLOAT))
                atol = 1e-2 * imprecision_factor * mindim
                rtol = 1e-2 * imprecision_factor

                if order == lib.enums.CblasRowMajor:
                    continue
                    #TODO: ROWMAJOR IMPLEMENTATION (LAPACK->LAPACKE)

                C = okcctx.CDenseMatrixContext(lib, mindim, mindim, order)
                x = okcctx.CVectorContext(lib, mindim, random=True)
                pivot = okcctx.CIntVectorContext(lib, mindim)
                hdl = okcctx.CDenseLapackContext(lib)

                with C, x, pivot, hdl as hdl:
                    # populate L
                    C.py *= 0
                    C.py += C_test
                    C.sync_to_c()

                    pysol = np.linalg.solve(C_test, x.py)

                    assert NO_ERR( lib.lapack_solve_LU(hdl, C.c, x.c, pivot.c) )
                    x.sync_to_py()
                    assert VEC_EQ( x.py, pysol, atol * mindim**0.5, rtol )

    def test_lapack_cholesky(self):
        A = self.Q_test.dot(np.diag(self.eigs).dot(self.Q_test.T))
        xrand = np.random.random(A.shape[0])
        LLT = np.linalg.cholesky(A)
        sol = np.linalg.solve(A, xrand)

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                nn = A.shape[0]
                imprecision_factor = 5**(int(lib.GPU) + int(lib.FLOAT))
                atol = 1e-2 * imprecision_factor * nn
                rtol = 1e-2 * imprecision_factor

                L = okcctx.CDenseMatrixContext(lib, nn, nn, order, value=A)
                x = okcctx.CVectorContext(lib, nn, value=xrand)
                hdl = okcctx.CDenseLapackContext(lib)

                with L, x, hdl as hdl:
                    assert NO_ERR(lib.lapack_cholesky_decomp(hdl, L.c))
                    assert NO_ERR( lib.linalg_cholesky_svx(hdl, L.c, x.c) )
                    x.sync_to_py()
                    assert VEC_EQ(x.py, sol, atol, rtol)

class DenseLinalgTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(DenseLinsysLibs())
        self.A_test = defs.A_test_gen()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_LAYOUTS = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS)

    def test_cholesky(self):
        m, n = defs.shape()
        mindim = min(m, n)

        # build decently conditioned symmetric matrix
        AA_test = self.A_test.T.dot(self.A_test)[:mindim, :mindim]
        AA_test /= np.linalg.norm(AA_test) * mindim**0.5
        for i in xrange(mindim):
            # diagonal entries ~ 1 to keep condition number reasonable
            AA_test[i, i] /= 10**np.log(mindim)
            AA_test[i, i] += 1
            # upper triangle = 0
            for j in xrange(mindim):
                if j > i:
                    AA_test[i, j] *= 0
        AA_test += AA_test.T

        x_rand = np.random.rand(mindim)
        pysol = np.linalg.solve(AA_test, x_rand)
        pychol = np.linalg.cholesky(AA_test)

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                imprecision_factor = 5**(int(lib.GPU) + int(lib.FLOAT))
                atol = 1e-2 * imprecision_factor * mindim
                rtol = 1e-2 * imprecision_factor

                L = okcctx.CDenseMatrixContext(lib, mindim, mindim, order)
                x = okcctx.CVectorContext(lib, mindim, random=True)
                hdl = okcctx.CDenseLinalgContext(lib)

                with L, x, hdl as hdl:
                    # populate L
                    L.py *= 0
                    L.py += AA_test
                    L.sync_to_c()

                    # cholesky factorization
                    assert NO_ERR( lib.linalg_cholesky_decomp(hdl, L.c) )
                    L.sync_to_py()

                    for i in xrange(mindim):
                        for j in xrange(mindim):
                            if j > i:
                                L.py[i, j] *= 0

                    assert VEC_EQ(
                            L.py.dot(x_rand), pychol.dot(x_rand), atol, rtol )

                    # cholesky solve
                    assert NO_ERR( lib.linalg_cholesky_svx(hdl, L.c, x.c) )
                    x.sync_to_py()
                    assert VEC_EQ( x.py, pysol, atol * mindim**0.5, rtol )

    def test_row_squares(self):
        m, n = defs.shape()
        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, ATOLM, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 1, 5)

                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)
                c = okcctx.CVectorContext(lib, n)
                r = okcctx.CVectorContext(lib, m)

                with A, c, r:
                    py_rows = [A.py[i, :].dot(A.py[i, :]) for i in xrange(m)]
                    py_cols = [A.py[:, j].dot(A.py[:, j]) for j in xrange(n)]

                    # C: calculate row squares
                    assert NO_ERR( lib.linalg_matrix_row_squares(
                            lib.enums.CblasNoTrans, A.c, r.c) )
                    r.sync_to_py()
                    assert VEC_EQ( r.py, py_rows, ATOLM, RTOL )

                    # C: calculate column squares
                    assert NO_ERR( lib.linalg_matrix_row_squares(
                            lib.enums.CblasTrans, A.c, c.c) )
                    c.sync_to_py()
                    assert VEC_EQ( c.py, py_cols, ATOLN, RTOL )

    def test_broadcast(self):
        m, n = defs.shape()

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, ATOLM, _, _ = CUSTOM_TOLS(lib, m, n, 1, 5)

                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)
                d = okcctx.CVectorContext(lib, m, random=True)
                e = okcctx.CVectorContext(lib, n, random=True)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, m)

                with A, d, e, x, y, hdl as hdl:
                    # A = A * diag(E)
                    assert NO_ERR( lib.linalg_matrix_broadcast_vector(
                            A.c, e.c, lib.enums.OkTransformScale,
                            lib.enums.CblasRight) )

                    assert NO_ERR( lib.blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    AEx = A.py.dot(e.py * x.py)
                    assert VEC_EQ( y.py, AEx, ATOLM, RTOL )

                    # A = diag(D) * A
                    assert NO_ERR( lib.linalg_matrix_broadcast_vector(
                            A.c, d.c, lib.enums.OkTransformScale, lib.enums.CblasLeft) )
                    assert NO_ERR( lib.blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    DAEx = d.py * AEx
                    assert VEC_EQ( y.py, DAEx, ATOLM, RTOL )

                    # A += 1e'
                    assert NO_ERR( lib.linalg_matrix_broadcast_vector(
                            A.c, e.c, lib.enums.OkTransformAdd, lib.enums.CblasRight) )
                    assert NO_ERR( lib.blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    A_updatex = DAEx + np.ones(m) * e.py.dot(x.py)
                    assert VEC_EQ( y.py, A_updatex, ATOLM, RTOL )

                    # A += d1'
                    assert NO_ERR( lib.linalg_matrix_broadcast_vector(
                            A.c, d.c, lib.enums.OkTransformAdd, lib.enums.CblasLeft) )
                    assert NO_ERR( lib.blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    A_updatex += d.py * sum(x.py)
                    assert VEC_EQ( y.py, A_updatex, ATOLM, RTOL )

    def test_reduce(self):
        m, n = defs.shape()

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, ATOLM, ATOLN, _ = CUSTOM_TOLS(lib, m, n, 1, 5)

                hdl = okcctx.CDenseLinalgContext(lib)
                A = okcctx.CDenseMatrixContext(lib, m, n, order, value=self.A_test)
                d = okcctx.CVectorContext(lib, m)
                e = okcctx.CVectorContext(lib, n)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, m)
                with A, d, e, x, y, hdl as hdl:
                    # min - reduce columns
                    colmin = np.min(A.py, 0)
                    assert NO_ERR( lib.linalg_matrix_reduce_min(
                            e.c, A.c, lib.enums.CblasLeft) )
                    e.sync_to_py()
                    assert VEC_EQ( e.py, colmin, ATOLN, RTOL )

                    # min - reduce rows
                    rowmin = np.min(A.py, 1)
                    assert NO_ERR( lib.linalg_matrix_reduce_min(
                            d.c, A.c, lib.enums.CblasRight) )
                    d.sync_to_py()
                    assert VEC_EQ( d.py, rowmin, ATOLM, RTOL )

                    # max - reduce columns
                    colmax = np.max(A.py, 0)
                    assert NO_ERR( lib.linalg_matrix_reduce_max(
                            e.c, A.c, lib.enums.CblasLeft) )
                    e.sync_to_py()
                    assert VEC_EQ( e.py, colmax, ATOLN, RTOL )

                    # max - reduce rows
                    rowmax = np.max(A.py, 1)
                    assert NO_ERR( lib.linalg_matrix_reduce_max(
                            d.c, A.c, lib.enums.CblasRight) )
                    d.sync_to_py()
                    assert VEC_EQ( d.py, rowmax, ATOLM, RTOL )

                    # indmin - reduce columns
                    with okcctx.CIndvectorContext(lib, n) as idx:
                        assert NO_ERR( lib.linalg_matrix_reduce_indmin(
                                idx.c, e.c, A.c, lib.enums.CblasLeft) )
                        idx.sync_to_py()
                        calcmin = np.array([A.py[idx.py[i], i] for i in xrange(n)])
                        colmin = np.min(A.py, 0)
                        assert VEC_EQ( calcmin, colmin, ATOLN, RTOL )

                    # indmin - reduce rows
                    with okcctx.CIndvectorContext(lib, m) as idx:
                        assert NO_ERR( lib.linalg_matrix_reduce_indmin(
                                idx.c, d.c, A.c, lib.enums.CblasRight) )
                        idx.sync_to_py()
                        calcmin = np.array([A.py[i, idx.py[i]] for i in xrange(m)])
                        rowmin = np.min(A.py, 1)
                        assert VEC_EQ( calcmin, rowmin, ATOLM, RTOL )
