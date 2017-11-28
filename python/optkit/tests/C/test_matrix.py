from optkit.compat import *

import os
import numpy as np
import itertools
import unittest

from optkit.libs import enums
from optkit.libs.linsys import DenseLinsysLibs
from optkit.tests import defs
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

STANDARD_TOLS = statements.standard_tolerances

class MatrixTestCase(unittest.TestCase):
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

    def test_alloc(self):
        m, n = defs.shape()

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                A = lib.matrix(0, 0, 0, None, order)
                assert ( A.size1 == 0 )
                assert ( A.size2 == 0 )
                assert ( A.ld == 0 )
                assert ( A.order == order )

                # calloc
                def m_alloc(): return lib.matrix_calloc(A, m, n, order)
                def m_free(): return lib.matrix_free(A)
                with okcctx.CVariableContext(m_alloc, m_free):
                    assert ( A.size1 == m )
                    assert ( A.size2 == n )
                    if order == lib.enums.CblasRowMajor:
                        assert ( A.ld == n )
                    else:
                        assert ( A.ld == m )
                    assert ( A.order == order )
                    if not lib.GPU:
                        for i in xrange(m * n):
                            assert( A.data[i] == 0 )

    def test_io(self):
        m, n = defs.shape()
        A_rand = self.A_test

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                TOL = 10**(-(11 - 4 * lib.FLOAT - 1 * lib.GPU))

                A = okcctx.CDenseMatrixContext(lib, m, n, order)
                Z = okcctx.CDenseMatrixContext(lib, m, n, order)
                with A, Z:

                    # memcpy_am
                    # set A_py to A_rand. overwrite A_py with zeros from A
                    A.py += A_rand
                    assert NO_ERR( lib.matrix_memcpy_am(A.pyptr, A.cptr, order) )
                    assert VEC_EQ( 0, A.py, 0, 0)

                    # memcpy_ma
                    A.py += A_rand
                    assert NO_ERR( lib.matrix_memcpy_ma(A.cptr, A.pyptr, order) )
                    A.py *= 0
                    assert NO_ERR( lib.matrix_memcpy_am(A.pyptr, A.cptr, order) )
                    assert VEC_EQ( A.py, A_rand, TOL, TOL )

                    # memcpy_mm
                    assert NO_ERR( lib.matrix_memcpy_mm(Z.cptr, A.cptr, order) )
                    assert NO_ERR( lib.matrix_memcpy_am(Z.pyptr, Z.cptr, order) )
                    assert VEC_EQ( Z.py, A.py, TOL, TOL )

                    # view_array
                    if not lib.GPU:
                        A.py *= 0
                        order_B = (
                                lib.enums.CblasRowMajor
                                if A_rand.flags.c_contiguous
                                else lib.enums.CblasColMajor
                        )
                        B = lib.matrix(0, 0, 0, None, order_B)
                        assert NO_ERR( lib.matrix_view_array(
                                B, A_rand.ctypes.data_as(lib.ok_float_p), m, n,
                                order_B) )
                        assert NO_ERR( lib.matrix_memcpy_am(A.pyptr, B, order) )
                        assert VEC_EQ( A.py, A_rand, TOL, TOL )

                    # set_all
                    val = 2
                    assert NO_ERR( lib.matrix_set_all(A.cptr, val) )
                    assert NO_ERR( lib.matrix_memcpy_am(A.pyptr, A.cptr, order) )
                    assert VEC_EQ( val, A.py, TOL, TOL )

    def test_slicing(self):
        """ matrix slicing tests """
        m, n = defs.shape()

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                TOL = 10**(-7 - 2 * lib.FLOAT - 1 * lib.GPU)

                A = okcctx.CDenseMatrixContext(lib, m, n, order)
                with A:
                    A.py += self.A_test
                    A.sync_to_c()

                    # submatrix
                    m0 = int(m / 4)
                    n0 = int(n / 4)
                    msub = int(m / 2)
                    nsub = int(n / 2)

                    Asub = lib.matrix(0, 0, 0, None, order)
                    Asub_py, Asub_ptr = okcctx.gen_py_matrix(lib, msub, nsub, order)

                    assert NO_ERR( lib.matrix_submatrix(Asub, A.c, m0, n0, msub, nsub) )
                    assert NO_ERR( lib.matrix_memcpy_am(Asub_ptr, Asub, order) )
                    A_py_sub = A.py[m0 : m0+msub, n0 : n0+nsub]
                    assert VEC_EQ( Asub_py, A_py_sub, TOL, TOL )

                    # row
                    v = lib.vector(0, 0, None)
                    v_py, v_ptr = okcctx.gen_py_vector(lib, n)
                    assert NO_ERR( lib.matrix_row(v, A.c, m0) )
                    assert NO_ERR( lib.vector_memcpy_av(v_ptr, v, 1) )
                    assert VEC_EQ( A.py[m0, :], v_py, TOL, TOL )

                    # column
                    v_py, v_ptr = okcctx.gen_py_vector(lib, m)
                    assert NO_ERR( lib.matrix_column(v, A.c, n0) )
                    assert NO_ERR( lib.vector_memcpy_av(v_ptr, v, 1) )
                    assert VEC_EQ( A.py[: , n0], v_py, TOL, TOL )

                    # diagonal
                    v_py, v_ptr = okcctx.gen_py_vector(lib, min(m, n))
                    assert NO_ERR( lib.matrix_diagonal(v, A.c) )
                    assert NO_ERR( lib.vector_memcpy_av(v_ptr, v, 1) )
                    assert VEC_EQ( np.diag(A.py), v_py, TOL, TOL )

    def test_math(self):
        """ matrix math tests """
        m, n = defs.shape()
        A_rand = self.A_test

        for lib, order in self.LIBS_LAYOUTS:
            with lib as lib:
                RTOL, _, _, ATOLMN = STANDARD_TOLS(lib, m, n, modulate_gpu=1)

                A = okcctx.CDenseMatrixContext(lib, m, n, order)
                with A:
                    # set A, A_py to A_rand
                    A.py += A_rand
                    A.sync_to_c()

                    # scale: A = alpha * A
                    alpha = np.random.random()
                    A_rand *= alpha
                    assert NO_ERR( lib.matrix_scale(A.c, alpha) )
                    A.sync_to_py()
                    assert VEC_EQ( A.py, A_rand, ATOLMN, RTOL )

                    # scale_left: A = diag(d) * A
                    with okcctx.CVectorContext(lib, m, random=True) as d:
                        for i in xrange(m):
                            A_rand[i, :] *= d.py[i]
                        assert NO_ERR( lib.matrix_scale_left(A.c, d.c) )
                        A.sync_to_py()
                        assert VEC_EQ( A.py, A_rand, ATOLMN, RTOL )

                    # scale_right: A = A * diag(e)
                    with okcctx.CVectorContext(lib, n, random=True) as e:
                        for j in xrange(n):
                            A_rand[:, j] *= e.py[j]
                        assert NO_ERR( lib.matrix_scale_right(A.c, e.c) )
                        A.sync_to_py()
                        assert VEC_EQ( A.py, A_rand, ATOLMN, RTOL )

                    # abs: A_ij = abs(A_ij)
                    A_rand -= (A_rand.max() - A_rand.min()) / 2
                    A.py *= 0
                    A.py += A_rand
                    A_rand = np.abs(A_rand)
                    A.sync_to_c()
                    assert NO_ERR( lib.matrix_abs(A.c) )
                    A.sync_to_py()
                    assert VEC_EQ( A.py, A_rand, ATOLMN, RTOL )

                    # pow
                    p = 3 * np.random.random()
                    A_rand **= p
                    assert NO_ERR( lib.matrix_pow(A.c, p) )
                    A.sync_to_py()
                    assert VEC_EQ( A.py, A_rand, ATOLMN, RTOL )

