from optkit.compat import *

import os
import numpy as np
import scipy.sparse as sp
import ctypes as ct
import itertools
import unittest

from optkit.libs import enums
from optkit.libs.linsys import SparseLinsysLibs
from optkit.tests import defs
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances

class SparseLibsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = SparseLinsysLibs()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_libs_exist(self):
        libs = []
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            libs.append(self.libs.get(single_precision=single_precision,
                                      gpu=gpu))
        assert any(libs)

    def test_lib_types(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                continue

            assert ('ok_int_p' in dir(lib))
            assert ('ok_float_p' in dir(lib))
            assert ('sparse_matrix' in dir(lib))
            assert ('sparse_matrix_p' in dir(lib))

    def test_sparse_handle(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                continue

            hdl = ct.c_void_p()
            assert NO_ERR( lib.sp_make_handle(ct.byref(hdl)) )
            assert NO_ERR( lib.sp_destroy_handle(hdl) )

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
                print("sparselib version", version)

class SparseMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(SparseLinsysLibs())

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.LIBS_LAYOUTS = itertools.product(
                self.libs, enums.OKEnums.MATRIX_ORDERS)

    def test_allocate(self):
        shape = (m, n) = defs.shape()
        nnz = int(0.05 * m * n)

        for lib, layout in self.LIBS_LAYOUTS:
            with lib as lib:
                A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, layout)

                # rowmajor: calloc, free
                def spm_alloc(): return lib.sp_matrix_calloc(A, m, n, nnz, layout)
                def spm_free(): return lib.sp_matrix_free(A)
                with okcctx.CVariableContext(spm_alloc, spm_free):
                    assert ( A.size1 == m )
                    assert ( A.size2 == n )
                    assert ( A.nnz == nnz )
                    dim = m if layout == lib.enums.CblasRowMajor else n
                    assert ( A.ptrlen == dim + 1 )
                    assert ( A.order == layout )
                    if not lib.GPU:
                        assert all([A.val[i] == 0 for i in xrange(2 * nnz)])
                        assert all([A.ind[i] == 0 for i in xrange(2 * nnz)])
                        assert all([(A.ptr[i] == 0) for i in xrange(2 + m + n)])

                assert ( A.size2 == 0 )
                assert ( A.nnz == 0 )
                assert ( A.ptrlen == 0 )


    def test_io(self):
        shape = (m, n) = defs.shape()
        x_rand = np.random.random(n)
        A_test = defs.A_test_sparse_gen()
        B_test = A_test * (1 + np.random.random((m, n)))

        for lib, layout in self.LIBS_LAYOUTS:
            with lib as lib:
                A = okcctx.CSparseMatrixContext(lib, A_test, layout)
                B = okcctx.CSparseMatrixContext(lib, B_test, layout)
                hdl = okcctx.CSparseLinalgContext(lib)
                RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

                with A, B, hdl as hdl:
                    # sparse handle
                    # Test sparse copy optkit->python
                    # A_ * x != A_c * x (calloc to zero)
                    val, ind, ptr = A.pyptr
                    assert NO_ERR( lib.sp_matrix_memcpy_am(val, ind, ptr, A.c) )
                    Ax_py = A.py.dot(x_rand)
                    Ax_c = A.sparse * x_rand
                    assert not VEC_EQ( Ax_py, Ax_c, ATOLM, RTOL )
                    assert VEC_EQ( 0, Ax_c, ATOLM, 0)

                    # Test sparse copy python->optkit
                    # B_py -> B_c -> B_py
                    # B_py * x == B_ * x
                    val, ind, ptr = B.pyptr
                    assert NO_ERR( lib.sp_matrix_memcpy_ma(hdl, B.c, val, ind, ptr) )
                    assert NO_ERR( lib.sp_matrix_memcpy_am(val, ind, ptr, B.c) )
                    Bx_py = B.py.dot(x_rand)
                    Bx_c = B.sparse * x_rand
                    assert VEC_EQ( Bx_py, Bx_c, ATOLM, RTOL )

                    # Test sparse copy optkit->optkit
                    # B_c -> A_c -> A_py
                    # B_py * x == A_py * x
                    val, ind, ptr = A.pyptr
                    assert NO_ERR( lib.sp_matrix_memcpy_mm(A.c, B.c) )
                    assert NO_ERR( lib.sp_matrix_memcpy_am(val, ind, ptr, A.c) )
                    Ax = A.sparse * x_rand
                    Bx = B.sparse * x_rand
                    assert VEC_EQ( Ax, Bx, ATOLM, RTOL )

                    # Test sparse value copy optkit->python
                    # A_py *= 0
                    # A_c -> A_py (values)
                    # B_py * x == A_py * x (still)
                    A.sparse *= 0
                    val, _, _ = A.pyptr
                    assert NO_ERR( lib.sp_matrix_memcpy_vals_am(val, A.c) )
                    Ax = A.sparse * x_rand
                    Bx = B.sparse * x_rand
                    assert VEC_EQ( Ax, Bx, ATOLM, RTOL )

                    # Test sparse value copy python->optkit
                    # A_py *= 2; A_py -> A_c; A_py *= 0; A_c -> A_py
                    # 2 * B_py * x == A_py * x
                    A.sparse *= 2
                    assert NO_ERR( lib.sp_matrix_memcpy_vals_ma(hdl, A.c, val) )
                    A.sparse *= 0
                    assert NO_ERR( lib.sp_matrix_memcpy_vals_am(val, A.c) )
                    Ax = A.sparse * x_rand
                    Bx = B.sparse * x_rand
                    assert VEC_EQ( Ax, 2 * Bx, ATOLM, RTOL )

                    # Test sparse value copy optkit->optkit
                    # A_c -> B_c -> B_py
                    # B_py * x == A_py * x
                    val, _, _ = B.pyptr
                    assert NO_ERR( lib.sp_matrix_memcpy_vals_mm(B.c, A.c) )
                    assert NO_ERR( lib.sp_matrix_memcpy_vals_am(val, B.c) )
                    Ax = A.sparse * x_rand
                    Bx = B.sparse * x_rand
                    assert VEC_EQ( Ax, Bx, ATOLM, RTOL )

    def test_multiply(self):
        shape = (m, n) = defs.shape()
        A_test = defs.A_test_sparse_gen()

        for lib, layout in self.LIBS_LAYOUTS:
            with lib as lib:
                A = okcctx.CSparseMatrixContext(lib, A_test, layout)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, m)
                hdl = okcctx.CSparseLinalgContext(lib)
                RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n)

                with A, x, y, hdl as hdl:
                    A.sync_to_c(hdl)

                    # y = Ax, Py vs. C
                    assert NO_ERR( lib.sp_blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1., A.c, x.c, 0., y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( A.py.dot(x.py), y.py, ATOLM, RTOL )

                    # x = A'y, Py vs. C
                    assert NO_ERR( lib.sp_blas_gemv(
                            hdl, lib.enums.CblasTrans, 1, A.c, y.c, 0, x.c) )
                    x.sync_to_py()
                    assert VEC_EQ( A.py.T.dot(y.py), x.py, ATOLN, RTOL )

                    # y = alpha Ax + beta y, Py vs. C
                    alpha = np.random.random()
                    beta = np.random.random()

                    result = alpha * A.py.dot(x.py) + beta * y.py
                    assert NO_ERR( lib.sp_blas_gemv(
                            hdl, lib.enums.CblasNoTrans, alpha, A.c, x.c, beta, y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( result, y.py, ATOLM, RTOL )

    def test_elementwise_transformations(self):
        shape = (m, n) = defs.shape()
        A_test = defs.A_test_sparse_gen()

        for lib, layout in self.LIBS_LAYOUTS:
            with lib as lib:
                A = okcctx.CSparseMatrixContext(lib, A_test, layout)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, m)
                hdl = okcctx.CSparseLinalgContext(lib)
                RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

                with A, x, y, hdl as hdl:
                    A.sync_to_c(hdl)
                    amax = A.sparse.data.max()

                    # abs
                    # make A_py mixed sign, load to A_c. then, A = abs(A)
                    A.sparse.data -= (amax / 2.)
                    A.py *= 0
                    A.py += np.abs(A.sparse.toarray())

                    A.sync_to_c(hdl)
                    assert NO_ERR( lib.sp_matrix_abs(A.c) )
                    assert NO_ERR( lib.sp_blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( A.py.dot(x.py), y.py, ATOLM, RTOL )

                    # pow
                    # A is nonnegative from previous step. set A_ij = A_ij ^ p
                    p = 3 * np.random.random()
                    A.py **= p
                    assert NO_ERR( lib.sp_matrix_pow(A.c, p) )
                    assert NO_ERR( lib.sp_blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( A.py.dot(x.py), y.py, ATOLM, RTOL )

                    # scale
                    # A = alpha * A
                    alpha = -1 + 2 * np.random.random()
                    A.py *= alpha
                    assert NO_ERR( lib.sp_matrix_scale(A.c, alpha) )
                    assert NO_ERR( lib.sp_blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    assert VEC_EQ( A.py.dot(x.py), y.py, ATOLM, RTOL )

    def test_diagonal_scaling(self):
        shape = (m, n) = defs.shape()
        A_test = defs.A_test_sparse_gen()

        for lib, layout in self.LIBS_LAYOUTS:
            with lib as lib:
                A = okcctx.CSparseMatrixContext(lib, A_test, layout)
                x = okcctx.CVectorContext(lib, n, random=True)
                y = okcctx.CVectorContext(lib, m)
                d = okcctx.CVectorContext(lib, m, random=True)
                e = okcctx.CVectorContext(lib, n, random=True)

                hdl = okcctx.CSparseLinalgContext(lib)
                RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

                with A, x, y, d, e, hdl as hdl:
                    A.sync_to_c(hdl)
                    amax = A.sparse.data.max()

                    # scale_left: A = diag(d) * A
                    # (form diag (d) * A * x, compare Py vs. C)
                    assert NO_ERR( lib.sp_matrix_scale_left(hdl, A.c, d.c) )
                    assert NO_ERR( lib.sp_blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    result = d.py * A.py.dot(x.py)
                    assert VEC_EQ( result, y.py, ATOLM, RTOL )

                    # scale_right: A = A * diag(e)
                    # (form diag (d) * A * diag(e) * x, compare Py vs. C)
                    assert NO_ERR( lib.sp_matrix_scale_right(hdl, A.c, e.c) )
                    assert NO_ERR( lib.sp_blas_gemv(
                            hdl, lib.enums.CblasNoTrans, 1, A.c, x.c, 0, y.c) )
                    y.sync_to_py()
                    result = d.py * A.py.dot(e.py * x.py)
                    assert VEC_EQ( result, y.py, ATOLM, RTOL )
